"""
Task 1 Grader — Glossary Anchoring.

Deterministic. No LLM. No semantic similarity.
Exact string match against pre-annotated answer keys.
"""

from typing import Tuple


class GlossaryGrader:
    """
    Grades ManuscriptAction for Task 1 (Glossary Anchoring).

    Scoring:
    - Exact correct answer:         1.00
    - Partial credit option:        0.40
    - Wrong answer:                 0.00
    - Invalid option (not in list): 0.00
    """

    FULL_CREDIT = 1.00
    PARTIAL_CREDIT = 0.40
    NO_CREDIT = 0.00

    def grade(
        self,
        selected_option: str,
        correct_answer: str,
        candidate_options: list,
        partial_credit_indices: list,
    ) -> Tuple[float, str]:
        """
        Grade a single glossary anchoring decision.

        Returns:
            (reward: float, feedback: str)
        """
        # Validate option is in candidate list
        if selected_option not in candidate_options:
            return (
                self.NO_CREDIT,
                f"Invalid selection. Must choose from the provided options exactly as written."
            )

        # Full credit
        if selected_option.strip() == correct_answer.strip():
            return (
                self.FULL_CREDIT,
                f"Correct. '{selected_option}' is the accurate domain-specific interpretation."
            )

        # Partial credit
        selected_index = candidate_options.index(selected_option)
        if selected_index in partial_credit_indices:
            return (
                self.PARTIAL_CREDIT,
                f"Partially correct. '{selected_option}' is related but not the most precise "
                f"domain-specific meaning. The best answer was: '{correct_answer}'."
            )

        # No credit
        return (
            self.NO_CREDIT,
            f"Incorrect. '{selected_option}' is not the right domain-specific meaning here. "
            f"The correct answer was: '{correct_answer}'."
        )
