"""
SanskritEnv client — importable by training scripts and notebooks.

Usage:
    from client import SanskritEnv
    from models import ManuscriptAction

    with SanskritEnv(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset(task_id="glossary_anchoring")
        obs = result.observation
        result = env.step(ManuscriptAction(
            selected_option=obs.candidate_options[0]
        ))
        print(result.reward)
"""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import ManuscriptAction, ManuscriptObservation, ManuscriptState


class SanskritEnv(EnvClient[ManuscriptAction, ManuscriptObservation, ManuscriptState]):

    def _step_payload(self, action: ManuscriptAction) -> dict:
        return {
            "selected_option": action.selected_option,
            "confidence": action.confidence,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = ManuscriptObservation(
            task_id=obs_data.get("task_id", ""),
            episode_id=obs_data.get("episode_id", ""),
            source_text_iast=obs_data.get("source_text_iast", ""),
            source_text_devanagari=obs_data.get("source_text_devanagari", ""),
            english_context=obs_data.get("english_context", ""),
            domain=obs_data.get("domain", ""),
            target_term_iast=obs_data.get("target_term_iast"),
            compound_iast=obs_data.get("compound_iast"),
            active_glossary=obs_data.get("active_glossary"),
            verses_so_far=obs_data.get("verses_so_far"),
            current_verse_num=obs_data.get("current_verse_num"),
            decision_prompt=obs_data.get("decision_prompt", ""),
            candidate_options=obs_data.get("candidate_options", []),
            step_reward=obs_data.get("step_reward", 0.0),
            cumulative_score=obs_data.get("cumulative_score", 0.0),
            feedback_message=obs_data.get("feedback_message", ""),
            consistency_history=obs_data.get("consistency_history"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> ManuscriptState:
        return ManuscriptState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            passage_id=payload.get("passage_id", ""),
            total_decisions=payload.get("total_decisions", 0),
            correct_decisions=payload.get("correct_decisions", 0),
            partial_decisions=payload.get("partial_decisions", 0),
            decision_history=payload.get("decision_history", []),
            consistency_map=payload.get("consistency_map", {}),
            is_complete=payload.get("is_complete", False),
        )
