"""
SanskritEnv Pydantic models — Action, Observation, State.

These types define the complete API contract for SanskritEnv.
All fields are documented. All types are JSON-serializable.
"""

from typing import List, Optional, Dict, Any, Union
from openenv.core.env_server import Action, Observation, State


class ManuscriptAction(Action):
    """
    The action an agent takes each step.

    The agent selects ONE option from the candidate_options list
    provided in the observation. The selected_option must exactly
    match one of the strings in candidate_options.
    """
    selected_option: str
    """The agent's chosen interpretation. Must match one candidate_options entry exactly."""

    confidence: float = 0.5
    """Agent's self-reported confidence, 0.0–1.0. Used for logging only, not graded."""

    reasoning: str = ""
    """Agent's explanation of its choice. Logged for analysis, not graded."""


class ManuscriptObservation(Observation):
    """
    What the agent observes at each step.

    Inherits: done: bool, reward: Optional[float] from Observation base.

    The agent must read source_text_iast, decision_prompt, and
    candidate_options, then return a ManuscriptAction with
    selected_option set to exactly one of the candidate_options strings.
    """
    # Episode metadata
    task_id: str
    """Which task: 'glossary_anchoring' | 'sandhi_resolution' | 'referential_coherence'"""

    episode_id: str
    """Unique identifier for this episode."""

    # Source text
    source_text_iast: str
    """The Sanskrit passage in IAST (International Alphabet of Sanskrit Transliteration)."""

    source_text_devanagari: str
    """The Sanskrit passage in Devanagari script."""

    english_context: str
    """Brief English description of the text's source and domain."""

    domain: str
    """Domain of the passage: 'ayurveda' | 'astronomy' | 'philosophy' | 'narrative'"""

    # For Task 1 (Glossary Anchoring)
    target_term_iast: Optional[str] = None
    """The specific term the agent must interpret (Task 1 only)."""

    active_glossary: Optional[Dict[str, str]] = None
    """Domain glossary entries for reference (Task 1 only)."""

    # For Task 2 (Sandhi Resolution)
    compound_iast: Optional[str] = None
    """The compound word to split (Task 2 only)."""

    # For Task 3 (Referential Coherence)
    verses_so_far: Optional[List[Dict[str, Any]]] = None
    """All verses seen so far in this episode (Task 3 only). List of dicts with keys: verse_num, iast, english."""

    current_verse_num: Optional[int] = None
    """Current verse number being processed (Task 3 only)."""

    # Decision interface
    decision_prompt: str
    """The specific question the agent must answer this step."""

    candidate_options: List[str]
    """
    Exactly 4 options. The agent must select one verbatim.
    Selecting a string not in this list returns reward=0 and done=True.
    """

    # Feedback
    step_reward: float = 0.0
    """Reward earned on the immediately preceding step. 0.0 on first step."""

    cumulative_score: float = 0.0
    """Running normalized score for this episode so far."""

    feedback_message: str = ""
    """Human-readable explanation of the previous step's reward."""

    # Consistency tracker (Task 3)
    consistency_history: Optional[List[Dict[str, str]]] = None
    """Prior checkpoint Q&A for this episode. Agent should maintain consistency."""


class ManuscriptState(State):
    """
    Episode-level state. Persists across all steps of an episode.

    Inherits: episode_id: Optional[str], step_count: int from State base.
    """
    task_id: str = ""
    """Active task identifier."""

    passage_id: str = ""
    """Which passage/episode is loaded."""

    total_decisions: int = 0
    """Total number of graded decisions in this episode."""

    correct_decisions: int = 0
    """Decisions scored as fully correct so far."""

    partial_decisions: int = 0
    """Decisions scored as partially correct so far."""

    decision_history: List[Dict[str, Any]] = []
    """Full trace: [{step, prompt, selected, correct, reward, timestamp}]"""

    consistency_map: Dict[str, str] = {}
    """For Task 3: maps referent labels to resolved antecedents across this episode."""

    is_complete: bool = False
    """True when all decisions in the episode have been made."""
