"""
Submission-safe inference script for SanskritEnv.

This script runs a single episode for a single task, emits only [START],
[STEP], and [END] lines to stdout, and uses the OpenAI client for all LLM
calls as required by the submission format.
"""

import asyncio
import logging
import os
import re
import sys
from contextlib import redirect_stderr
from io import StringIO
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from client import SanskritEnv
from models import ManuscriptAction


logging.getLogger("dotenv.main").setLevel(logging.ERROR)

with redirect_stderr(StringIO()):
    load_dotenv()


HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
API_BASE_URL = (os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1").strip()
MODEL_NAME = (os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct").strip()
LOCAL_IMAGE_NAME = (os.getenv("LOCAL_IMAGE_NAME") or "").strip()
LOCAL_BASE_URL = (os.getenv("SANSKRIT_ENV_URL") or "http://localhost:7860").strip()

BENCHMARK = "sanskrit-env"
SPACE_BASE_URL = "https://adityahars-sanskrit-env.hf.space"
RANDOM_SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 256
REQUEST_TIMEOUT = 90
EPISODES_PER_TASK = 5

TASK_SEQUENCE = [
    "glossary_anchoring",
    "sandhi_resolution",
    "samasa_classification",
    "referential_coherence",
    "manuscript_restoration",
    "full_manuscript_session",
]
VALID_TASKS = set(TASK_SEQUENCE)
TASK_LABELS = {
    "glossary_anchoring": "glossary anchoring (easy)",
    "sandhi_resolution": "sandhi resolution (medium)",
    "samasa_classification": "samasa classification (medium)",
    "referential_coherence": "referential coherence (hard)",
    "manuscript_restoration": "manuscript restoration (expert)",
    "full_manuscript_session": "full manuscript session (master)",
}
MAX_STEPS_BY_TASK = {
    "glossary_anchoring": 1,
    "sandhi_resolution": 1,
    "samasa_classification": 1,
    "referential_coherence": 7,
    "manuscript_restoration": 10,
    "full_manuscript_session": 20,
}

SYSTEM_PROMPT = """You are an expert Sanskrit manuscript interpreter.
Read the passage, question, and candidate options carefully.
Reply with exactly one candidate option copied verbatim.
Do not add explanations or extra text.
""".strip()


def _debug(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _single_line(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _clamp_score(value: Optional[float]) -> float:
    try:
        numeric = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    return min(max(numeric, 0.0), 1.0)


def _extract_completion_text(completion) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""

    content = getattr(choices[0].message, "content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
                continue

            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()

    return str(content or "").strip()


def build_task_plan(task_id: str, episodes_per_task: int = EPISODES_PER_TASK) -> List[str]:
    normalized_task_id = task_id if task_id in VALID_TASKS else TASK_SEQUENCE[0]
    return [normalized_task_id] * max(1, episodes_per_task)


def build_task_label(task_id: str) -> str:
    return TASK_LABELS.get(task_id, task_id.replace("_", " "))


def build_user_prompt(obs, rolling_memory: str) -> str:
    lines: List[str] = []

    if getattr(obs, "source_text_iast", ""):
        lines.append(f"Sanskrit (IAST): {obs.source_text_iast}")
    if getattr(obs, "source_text_devanagari", ""):
        lines.append(f"Devanagari: {obs.source_text_devanagari}")
    if getattr(obs, "english_context", ""):
        lines.append(f"Source context: {obs.english_context}")
    if getattr(obs, "domain", ""):
        lines.append(f"Domain: {obs.domain}")
    if getattr(obs, "target_term_iast", None):
        lines.append(f"Term to interpret: {obs.target_term_iast}")
    if getattr(obs, "compound_iast", None):
        label = "Compound to classify" if obs.task_id == "samasa_classification" else "Compound to split"
        lines.append(f"{label}: {obs.compound_iast}")

    verses = getattr(obs, "verses_so_far", None)
    if verses:
        lines.append("")
        lines.append("Verses in this passage:")
        for verse in verses:
            lines.append(f"[{verse['verse_num']}] IAST: {verse['iast']}")
            lines.append(f"English: {verse['english']}")

    if rolling_memory.strip():
        lines.append("")
        lines.append("Previous decisions:")
        lines.append(rolling_memory.strip())

    if getattr(obs, "step_reward", None) and obs.step_reward > 0:
        lines.append("")
        lines.append(f"Previous step reward: {obs.step_reward:.2f}")
    elif getattr(obs, "feedback_message", ""):
        lines.append("")
        lines.append(f"Feedback: {obs.feedback_message}")

    lines.append("")
    lines.append(f"Question: {getattr(obs, 'decision_prompt', '')}")
    lines.append("")
    lines.append("Options:")
    for index, option in enumerate(getattr(obs, "candidate_options", []) or [], start=1):
        lines.append(f"{index}. {option}")
    lines.append("")
    lines.append("Reply with exactly one option.")

    return "\n".join(lines)


def update_rolling_memory(rolling_memory: str, obs, selected_option: str) -> str:
    prompt = getattr(obs, "decision_prompt", "")
    if not prompt:
        return rolling_memory

    summary = f"{prompt.strip().rstrip('?')} -> {selected_option}"
    lines = [line for line in rolling_memory.splitlines() if line.strip()]
    lines.append(summary)
    return "\n".join(lines[-10:])


def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured")

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
        timeout=REQUEST_TIMEOUT,
    )

    text = _extract_completion_text(completion)
    if text:
        return text
    raise RuntimeError("empty model response")


def match_to_option(raw_answer: str, candidate_options: List[str]) -> str:
    if not candidate_options:
        raise RuntimeError("environment returned no candidate options")

    raw = (raw_answer or "").strip()
    if not raw:
        return candidate_options[0]

    for option in candidate_options:
        if raw == option:
            return option

    numeric_match = re.fullmatch(
        r"(?:option\s*)?[\[(]?([1-9]\d*)[\])\.:\-]?(?:\s+.*)?",
        raw,
        flags=re.IGNORECASE,
    )
    if numeric_match:
        option_index = int(numeric_match.group(1)) - 1
        if 0 <= option_index < len(candidate_options):
            return candidate_options[option_index]

    for option in candidate_options:
        if option.lower().startswith(raw.lower()[:30]):
            return option

    for option in candidate_options:
        if option.lower() in raw.lower():
            return option

    return candidate_options[0]


def choose_action(client: OpenAI, obs, rolling_memory: str) -> Tuple[str, str, Optional[str]]:
    candidate_options = getattr(obs, "candidate_options", []) or []
    if not candidate_options:
        raise RuntimeError("environment returned no candidate options")

    prompt = build_user_prompt(obs, rolling_memory)
    try:
        raw_answer = call_llm(client, SYSTEM_PROMPT, prompt)
        return match_to_option(raw_answer, candidate_options), raw_answer, None
    except Exception as exc:
        error = _single_line(str(exc)) or "model request failed"
        _debug(f"model fallback: {error}")
        return candidate_options[0], "", error


def _extract_step_error(obs, model_error: Optional[str]) -> Optional[str]:
    if model_error:
        return model_error

    feedback = _single_line(getattr(obs, "feedback_message", ""))
    lowered = feedback.lower()
    if feedback and ("error" in lowered or "invalid" in lowered or "not found" in lowered):
        return feedback

    return None


def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


async def create_env() -> SanskritEnv:
    if LOCAL_IMAGE_NAME:
        env = SanskritEnv(base_url=LOCAL_BASE_URL)
        try:
            await env.connect()
            return env
        except Exception as exc:
            raise RuntimeError(
                f"LOCAL_IMAGE_NAME is set, but no local environment is reachable at {LOCAL_BASE_URL}. "
                f"Start it manually with: docker run --rm -p 7860:7860 {LOCAL_IMAGE_NAME}"
            ) from exc

    env = SanskritEnv(base_url=SPACE_BASE_URL)
    await env.connect()
    return env


async def run_episode(
    env: SanskritEnv,
    client: OpenAI,
    task_id: str,
    seed: int,
    step_offset: int,
) -> Tuple[int, List[float], float, bool]:
    try:
        result = await env.reset(task_id=task_id, seed=seed)
    except asyncio.CancelledError:
        _debug(f"env.reset cancelled for task {task_id}")
        return 0, [], 0.0, False
    except Exception as exc:
        _debug(f"env.reset failed for task {task_id}: {_single_line(str(exc))}")
        return 0, [], 0.0, False

    observation = result.observation
    rolling_memory = ""
    rewards: List[float] = []
    steps_taken = 0

    for step in range(1, MAX_STEPS_BY_TASK[task_id] + 1):
        if bool(result.done or getattr(observation, "done", False)):
            break

        selected_option, raw_answer, model_error = choose_action(client, observation, rolling_memory)
        rolling_memory = update_rolling_memory(rolling_memory, observation, selected_option)
        global_step = step_offset + step

        try:
            result = await env.step(
                ManuscriptAction(
                    selected_option=selected_option,
                    confidence=0.8,
                    reasoning=raw_answer or model_error or "",
                )
            )
        except asyncio.CancelledError:
            steps_taken = step
            log_step(global_step, selected_option, 0.0, True, "environment step cancelled")
            break
        except Exception as exc:
            steps_taken = step
            log_step(global_step, selected_option, 0.0, True, _single_line(str(exc)) or "environment step failed")
            break

        observation = result.observation
        reward = getattr(observation, "step_reward", None)
        if reward is None:
            reward = result.reward
        reward = float(reward if reward is not None else 0.0)

        rewards.append(reward)
        steps_taken = step

        done = bool(result.done or getattr(observation, "done", False))
        step_error = _extract_step_error(observation, model_error)
        log_step(global_step, selected_option, reward, done, step_error)

        if done:
            break

    score = _clamp_score(getattr(observation, "cumulative_score", 0.0))
    if score == 0.0 and getattr(result, "reward", None) is not None:
        score = _clamp_score(result.reward)

    success = bool(result.done or getattr(observation, "done", False)) and score > 0.50
    return steps_taken, rewards, score, success


def log_score_summary(task_scores: dict[str, List[float]]) -> None:
    for task_id in TASK_SEQUENCE:
        scores = task_scores.get(task_id, [])
        if not scores:
            continue
        mean_score = sum(scores) / len(scores)
        score_text = ",".join(f"{score:.2f}" for score in scores)
        _debug(
            f"scores task={task_id} label={build_task_label(task_id)} episodes={len(scores)} "
            f"mean={mean_score:.4f} values={score_text}"
        )


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "missing")

    task_scores = {task_id: [] for task_id in TASK_SEQUENCE}
    stop_requested = False

    for task_index, task_id in enumerate(TASK_SEQUENCE):
        env: Optional[SanskritEnv] = None
        task_rewards: List[float] = []
        task_episode_scores: List[float] = []
        task_steps_taken = 0
        task_success = False
        task_score = 0.0
        task_plan = build_task_plan(task_id)

        log_start(task=build_task_label(task_id), env=BENCHMARK, model=MODEL_NAME)

        try:
            env = await create_env()

            for episode_index, planned_task_id in enumerate(task_plan):
                episode_steps, episode_rewards, episode_score, _ = await run_episode(
                    env=env,
                    client=client,
                    task_id=planned_task_id,
                    seed=RANDOM_SEED + task_index * EPISODES_PER_TASK + episode_index,
                    step_offset=task_steps_taken,
                )
                task_steps_taken += episode_steps
                task_rewards.extend(episode_rewards)
                task_episode_scores.append(episode_score)
                task_scores[task_id].append(episode_score)

            if task_episode_scores:
                task_score = sum(task_episode_scores) / len(task_episode_scores)
            task_success = len(task_episode_scores) == len(task_plan) and task_score > 0.50
            log_score_summary({task_id: task_episode_scores})
        except asyncio.CancelledError:
            _debug(f"task inference cancelled ({task_id})")
            if task_episode_scores:
                task_score = sum(task_episode_scores) / len(task_episode_scores)
            task_success = False
            stop_requested = True
        except Exception as exc:
            _debug(f"task inference error ({task_id}): {_single_line(str(exc))}")
            if task_episode_scores:
                task_score = sum(task_episode_scores) / len(task_episode_scores)
            task_success = False
        except BaseException as exc:
            _debug(f"task inference interrupted ({task_id}): {_single_line(str(exc)) or type(exc).__name__}")
            if task_episode_scores:
                task_score = sum(task_episode_scores) / len(task_episode_scores)
            task_success = False
            stop_requested = True
        finally:
            if env is not None:
                try:
                    await env.close()
                except BaseException as exc:
                    _debug(f"env.close failed for task {task_id}: {_single_line(str(exc))}")

            log_end(
                success=task_success,
                steps=task_steps_taken,
                score=task_score,
                rewards=task_rewards,
            )

        if stop_requested:
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
