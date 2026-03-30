"""
Task 2 Grader — Sandhi Resolution.

Deterministic. Exact IAST string match against annotated split table.
Sandhi splits must match the canonical form exactly.
"""

from typing import Tuple


class SandhiGrader:
    """
    Grades ManuscriptAction for Task 2 (Sandhi Resolution).

    Scoring:
    - Exact correct split:            1.00
    - Semantically adjacent option:   0.25  (same first component, wrong second)
    - Wrong split:                    0.00
    - Invalid option:                 0.00

    No partial credit indices for Sandhi — the split is either right or wrong.
    The 0.25 tier applies when options A and C are equivalent (as in t2_008).
    """

    FULL_CREDIT = 1.00
    ADJACENT_CREDIT = 0.25
    NO_CREDIT = 0.00

    def grade(
        self,
        selected_option: str,
        correct_answer: str,
        candidate_options: list,
        partial_credit_indices: list,
    ) -> Tuple[float, str]:
        """
        Grade a sandhi resolution decision.

        Returns:
            (reward: float, feedback: str)
        """
        if selected_option not in candidate_options:
            return (
                self.NO_CREDIT,
                "Invalid selection. Choose from the provided options exactly."
            )

        if selected_option.strip() == correct_answer.strip():
            return (
                self.FULL_CREDIT,
                f"Correct sandhi split. '{selected_option}' is the contextually accurate resolution."
            )

        selected_index = candidate_options.index(selected_option)
        if selected_index in partial_credit_indices:
            return (
                self.ADJACENT_CREDIT,
                f"Adjacent analysis. Your split is linguistically similar to the correct one. "
                f"Canonical answer: '{correct_answer}'."
            )

        return (
            self.NO_CREDIT,
            f"Incorrect split. '{selected_option}' does not resolve correctly in this context. "
            f"The correct split was: '{correct_answer}'."
        )
