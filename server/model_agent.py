import json
import hashlib
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from models import ManuscriptAction


SYSTEM_PROMPT = """You are an expert Sanskrit manuscript interpreter with deep knowledge of:
- Classical Sanskrit grammar, phonology, and sandhi rules
- Paninian grammar: samasa classification (Tatpurusha, Karmadharaya, Dvigu, Dvandva, Bahuvrihi, Avyayibhava)
- Ayurvedic, astronomical, philosophical, and narrative Sanskrit literature

For each step:
1. Think carefully about the passage and question.
2. Choose exactly one option from the provided list.
3. Output only the exact selected option text.

Rules:
- Return exactly one provided option, copied character-for-character.
- Do not add any explanation or extra text.
"""


DEFAULT_FREE_MODELS: List[Dict[str, str]] = [
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "label": "Llama 3.3 70B Instruct"},
    {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen 2.5 72B Instruct"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen 2.5 7B Instruct"},
    {"id": "google/gemma-2-9b-it", "label": "Gemma 2 9B Instruct"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B Instruct v0.3"},
    {"id": "HuggingFaceH4/zephyr-7b-beta", "label": "Zephyr 7B Beta"},
]

_MODEL_CATALOG_CACHE: Dict[str, Dict[str, Any]] = {}


def get_model_catalog(configured_models: str = "") -> List[Dict[str, str]]:
    """
    Build model options from HF_UI_MODELS env var if provided,
    otherwise use a curated free-tier default list.
    """
    if configured_models.strip():
        ids = [m.strip() for m in configured_models.split(",") if m.strip()]
        return [{"id": model_id, "label": model_id} for model_id in ids]

    return [dict(item) for item in DEFAULT_FREE_MODELS]


def _parse_router_error_text(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "unknown provider error"

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            if isinstance(err, str) and err.strip():
                return err.strip()
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    title_start = lowered.find("<title>")
    if title_start != -1:
        title_end = lowered.find("</title>", title_start)
        if title_end != -1:
            title = text[title_start + len("<title>"):title_end].strip()
            if title:
                return title

    return " ".join(text.split())[:220]


def _probe_model_availability(
    model_id: str,
    hf_token: str,
    router_url: str,
    request_timeout: int,
) -> tuple[bool, str]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Reply with OK only."},
            {"role": "user", "content": "OK"},
        ],
        "temperature": 0,
        "max_tokens": 4,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(router_url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            if 200 <= response.status < 300:
                return True, "ok"
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="ignore")
        parsed = _parse_router_error_text(err_text)
        return False, f"{exc.code}: {parsed}"
    except urllib.error.URLError as exc:
        return False, f"network: {exc.reason}"
    except Exception as exc:
        return False, f"error: {exc}"


def get_available_model_catalog(
    configured_models: str,
    hf_token: str,
    router_url: str,
    request_timeout: int,
    cache_ttl: int = 300,
) -> Dict[str, Any]:
    models = get_model_catalog(configured_models)
    if not models:
        return {
            "models": [],
            "unavailable_models": [],
            "availability_checked": False,
            "catalog_size": 0,
        }

    if not hf_token:
        return {
            "models": models,
            "unavailable_models": [],
            "availability_checked": False,
            "catalog_size": len(models),
        }

    safe_timeout = max(6, min(request_timeout, 30))
    ttl = max(30, cache_ttl)
    token_digest = hashlib.sha256(hf_token.encode("utf-8")).hexdigest()[:16]
    cache_key = f"{token_digest}|{router_url}|{configured_models.strip()}"

    now = time.time()
    cached = _MODEL_CATALOG_CACHE.get(cache_key)
    if cached and now - cached.get("ts", 0) < ttl:
        return cached["data"]

    available: List[Dict[str, str]] = []
    unavailable: List[Dict[str, str]] = []

    for model in models:
        ok, reason = _probe_model_availability(
            model_id=model["id"],
            hf_token=hf_token,
            router_url=router_url,
            request_timeout=safe_timeout,
        )
        if ok:
            available.append(model)
        else:
            unavailable.append(
                {
                    "id": model["id"],
                    "label": model["label"],
                    "reason": reason,
                }
            )

    data = {
        "models": available,
        "unavailable_models": unavailable,
        "availability_checked": True,
        "catalog_size": len(models),
    }

    _MODEL_CATALOG_CACHE[cache_key] = {"ts": now, "data": data}
    return data


def build_user_prompt(obs: Any, rolling_memory: str) -> str:
    lines: List[str] = []

    if getattr(obs, "source_text_iast", None):
        lines.append(f"Sanskrit (IAST): {obs.source_text_iast}")
    if getattr(obs, "source_text_devanagari", None):
        lines.append(f"Devanagari:      {obs.source_text_devanagari}")
    if getattr(obs, "english_context", None):
        lines.append(f"Source context:  {obs.english_context}")
    if getattr(obs, "domain", None):
        lines.append(f"Domain:          {obs.domain}")

    if getattr(obs, "target_term_iast", None):
        lines.append(f"Term to interpret: {obs.target_term_iast}")

    if getattr(obs, "compound_iast", None):
        label = "Compound to classify" if getattr(obs, "task_id", "") == "samasa_classification" else "Compound to split"
        lines.append(f"{label}: {obs.compound_iast}")

    verses = getattr(obs, "verses_so_far", None)
    if verses:
        lines.append("")
        lines.append("Verses in this passage:")
        for verse in verses:
            lines.append(f"  [{verse['verse_num']}] IAST:    {verse['iast']}")
            lines.append(f"       English: {verse['english']}")

    if rolling_memory.strip():
        lines.append("")
        lines.append("What you have established so far in this episode:")
        lines.append(rolling_memory.strip())

    if getattr(obs, "step_reward", None) and obs.step_reward > 0:
        lines.append("")
        lines.append(f"Your last answer was correct (reward: {obs.step_reward:.2f}).")
    elif getattr(obs, "step_reward", None) == 0.0 and getattr(obs, "feedback_message", None):
        lines.append("")
        lines.append(f"Feedback: {obs.feedback_message}")

    lines.append("")
    lines.append(f"Question: {obs.decision_prompt}")
    lines.append("")
    lines.append("Options (choose one exactly as written):")
    for i, option in enumerate(obs.candidate_options):
        lines.append(f"  {i + 1}. {option}")
    lines.append("")
    lines.append("Your answer (exact option text only):")

    return "\n".join(lines)


def update_rolling_memory(rolling_memory: str, obs: Any, selected_option: str) -> str:
    if not getattr(obs, "decision_prompt", None):
        return rolling_memory

    summary = f"- {obs.decision_prompt.strip().rstrip('?')} -> {selected_option}"
    lines = [line for line in rolling_memory.strip().split("\n") if line.strip()]
    lines.append(summary)
    if len(lines) > 10:
        lines = lines[-10:]
    return "\n".join(lines)


def match_to_option(raw_answer: str, candidate_options: List[str]) -> str:
    raw = (raw_answer or "").strip()

    for option in candidate_options:
        if raw == option:
            return option

    for option in candidate_options:
        if raw and option.lower().startswith(raw.lower()[:30]):
            return option

    for option in candidate_options:
        if option.lower() in raw.lower():
            return option

    return candidate_options[0] if candidate_options else ""


def _extract_router_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()

    return str(content).strip()


def call_hf_router(
    model_id: str,
    user_prompt: str,
    hf_token: str,
    router_url: str,
    temperature: float,
    max_tokens: int,
    retry_wait: int,
    request_timeout: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    for attempt in range(4):
        request = urllib.request.Request(router_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                text = _extract_router_text(response_data)
                if text:
                    return text
                raise RuntimeError("Model returned empty text.")
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="ignore")
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(retry_wait * (2 ** attempt))
                continue
            parsed = _parse_router_error_text(err_text)
            raise RuntimeError(f"HF router error {exc.code}: {parsed}")
        except urllib.error.URLError as exc:
            time.sleep(retry_wait * (2 ** attempt))
            if attempt == 3:
                raise RuntimeError(f"HF router network error: {exc.reason}")

    raise RuntimeError("HF router retries exhausted.")


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return dict(obs)


def run_model_episode(
    env: Any,
    task_id: str,
    model_id: str,
    hf_token: str,
    router_url: str,
    temperature: float,
    max_tokens: int,
    retry_wait: int,
    request_timeout: int,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    session_id = episode_id or f"model_{uuid.uuid4().hex}"
    obs = env.reset(seed=seed, episode_id=session_id, task_id=task_id)

    start = time.perf_counter()
    rolling_memory = ""
    steps: List[Dict[str, Any]] = []

    max_steps = 16
    while not obs.done and len(steps) < max_steps:
        user_prompt = build_user_prompt(obs, rolling_memory)
        raw_answer = call_hf_router(
            model_id=model_id,
            user_prompt=user_prompt,
            hf_token=hf_token,
            router_url=router_url,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_wait=retry_wait,
            request_timeout=request_timeout,
        )

        selected = match_to_option(raw_answer, obs.candidate_options)
        rolling_memory = update_rolling_memory(rolling_memory, obs, selected)

        next_obs = env.step(
            ManuscriptAction(
                selected_option=selected,
                confidence=0.8,
                reasoning=raw_answer,
            ),
            request_id=session_id,
        )

        steps.append(
            {
                "step": len(steps) + 1,
                "decision_prompt": obs.decision_prompt,
                "selected_option": selected,
                "step_reward": next_obs.step_reward,
                "cumulative_score": next_obs.cumulative_score,
                "feedback_message": next_obs.feedback_message,
            }
        )

        obs = next_obs

    if not obs.done:
        raise RuntimeError("Model run hit safety limit before finishing episode.")

    elapsed_seconds = round(time.perf_counter() - start, 3)
    final_score = obs.reward if obs.reward is not None else obs.cumulative_score

    return {
        "task_id": task_id,
        "model_id": model_id,
        "episode_id": session_id,
        "step_count": len(steps),
        "steps": steps,
        "final_score": round(float(final_score), 4),
        "elapsed_seconds": elapsed_seconds,
        "observation": _obs_to_dict(obs),
    }
