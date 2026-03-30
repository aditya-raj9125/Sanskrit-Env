"""
Task 3 Grader — Referential Coherence.

Deterministic. Checks antecedent identification against pre-annotated answers.
Also checks consistency with checkpoints established during the episode.

This is the HARD grader. It scores two things:
1. Final referential question answer (main score)
2. Consistency with intermediate checkpoints (bonus signal)
"""

from typing import Tuple, List, Dict


class CoherenceGrader:
    """
    Grades ManuscriptAction for Task 3 (Referential Coherence).

    The main question is worth 0.70 of the episode score.
    Each consistency checkpoint correctly answered is worth 0.10 (up to 0.30).
    Total episode score is normalized to 0.0–1.0.

    Main question scoring:
    - Correct antecedent: 0.70
    - Wrong antecedent:   0.00

    Per checkpoint:
    - Correct:  +0.10
    - Wrong:    +0.00 (no penalty — checkpoints are informational)
    """

    MAIN_CORRECT = 0.70
    CHECKPOINT_CORRECT = 0.10
    NO_CREDIT = 0.00

    def grade_final(
        self,
        selected_option: str,
        correct_answer: str,
        candidate_options: list,
    ) -> Tuple[float, str]:
        """Grade the final referential question."""

        if selected_option not in candidate_options:
            return (
                self.NO_CREDIT,
                "Invalid selection."
            )

        if selected_option.strip() == correct_answer.strip():
            return (
                self.MAIN_CORRECT,
                f"Correct antecedent identification. '{selected_option}' is the proper "
                f"referent in context."
            )

        return (
            self.NO_CREDIT,
            f"Incorrect antecedent. '{selected_option}' does not match the contextual referent. "
            f"The correct answer was: '{correct_answer}'."
        )

    def grade_checkpoint(
        self,
        selected_option: str,
        correct_answer: str,
        candidate_options: list,
    ) -> Tuple[float, str]:
        """Grade a mid-episode consistency checkpoint."""

        if selected_option not in candidate_options:
            return (self.NO_CREDIT, "Invalid selection at checkpoint.")

        if selected_option.strip() == correct_answer.strip():
            return (
                self.CHECKPOINT_CORRECT,
                f"Checkpoint correct. Reference '{selected_option}' is consistent."
            )

        return (
            self.NO_CREDIT,
            f"Checkpoint incorrect. Expected '{correct_answer}', got '{selected_option}'."
        )

    def compute_episode_score(
        self,
        final_reward: float,
        checkpoint_rewards: List[float],
    ) -> float:
        """
        Compute normalized final episode score.

        Max possible = 0.70 (final) + 0.10 * num_checkpoints
        Normalized to 0.0–1.0.
        """
        total = final_reward + sum(checkpoint_rewards)
        max_possible = self.MAIN_CORRECT + self.CHECKPOINT_CORRECT * len(checkpoint_rewards)
        if max_possible == 0:
            return 0.0
        return round(min(total / max_possible, 1.0), 4)
