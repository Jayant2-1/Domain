"""
Skill Engine — ELO-based adaptive rating system for DSA tutoring.

Implements a modified ELO rating system where the user competes against
question difficulty. Pure math, no DB, no side effects, fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# K-factor schedule
# ---------------------------------------------------------------------------

def get_k_factor(matches: int) -> float:
    """
    Return the K-factor based on the number of matches played.

    - matches < 10  → K = 40  (new users converge faster)
    - 10 <= matches < 30 → K = 32  (standard)
    - matches >= 30 → K = 24  (established users change slowly)
    """
    if matches < 10:
        return 40.0
    if matches < 30:
        return 32.0
    return 24.0


# ---------------------------------------------------------------------------
# Core ELO functions (stateless, deterministic)
# ---------------------------------------------------------------------------

def expected_score(user_rating: float, question_difficulty: float) -> float:
    """
    Compute the expected probability that the user answers correctly.

    E_u = 1 / (1 + 10^((R_q - R_u) / 400))

    Returns a float in (0, 1).
    """
    exponent = (question_difficulty - user_rating) / 400.0
    return 1.0 / (1.0 + math.pow(10.0, exponent))


def rating_delta(
    user_rating: float,
    question_difficulty: float,
    answered_correctly: bool,
    k_factor: float,
) -> float:
    """
    Compute the raw rating change (delta) for a single interaction.

    delta = K * (S - E_u)

    Where S = 1 if correct, S = 0 if incorrect.
    """
    expected = expected_score(user_rating, question_difficulty)
    actual = 1.0 if answered_correctly else 0.0
    return k_factor * (actual - expected)


def compute_new_rating(
    user_rating: float,
    question_difficulty: float,
    answered_correctly: bool,
    matches: int,
    k_override: float | None = None,
) -> Tuple[float, float]:
    """
    Compute the updated user rating after answering a question.

    Parameters
    ----------
    user_rating : float
        Current ELO rating of the user for the topic.
    question_difficulty : float
        Difficulty rating of the question.
    answered_correctly : bool
        Whether the user's answer was correct.
    matches : int
        Number of prior matches the user has played in this topic (for K-factor).
    k_override : float | None
        If provided, overrides the K-factor schedule (for testing / tuning).

    Returns
    -------
    tuple[float, float]
        (new_rating, delta) — the updated rating and the change applied.
    """
    k = k_override if k_override is not None else get_k_factor(matches)
    delta = rating_delta(user_rating, question_difficulty, answered_correctly, k)
    new_rating = user_rating + delta
    return new_rating, delta


# ---------------------------------------------------------------------------
# TopicRating — lightweight in-memory container for one (user, topic) pair
# ---------------------------------------------------------------------------

@dataclass
class TopicRating:
    """Holds current ELO state for a single topic."""
    rating: float = 1000.0
    matches: int = 0

    def update(
        self,
        question_difficulty: float,
        answered_correctly: bool,
        k_override: float | None = None,
    ) -> float:
        """
        Apply one interaction and return the delta.

        Mutates self.rating and self.matches in-place.
        """
        new_rating, delta = compute_new_rating(
            user_rating=self.rating,
            question_difficulty=question_difficulty,
            answered_correctly=answered_correctly,
            matches=self.matches,
            k_override=k_override,
        )
        self.rating = new_rating
        self.matches += 1
        return delta


# ---------------------------------------------------------------------------
# SkillProfile — aggregates TopicRatings across all topics for one user
# ---------------------------------------------------------------------------

@dataclass
class SkillProfile:
    """
    In-memory skill profile for a single user.

    Maps topic names to their TopicRating. No database dependency —
    this is a pure data container used by the service layer.
    """
    ratings: Dict[str, TopicRating] = field(default_factory=dict)

    def get_rating(self, topic: str) -> TopicRating:
        """Return the TopicRating for a topic, creating a default if absent."""
        if topic not in self.ratings:
            self.ratings[topic] = TopicRating()
        return self.ratings[topic]

    def update(
        self,
        topic: str,
        question_difficulty: float,
        answered_correctly: bool,
        k_override: float | None = None,
    ) -> Tuple[float, float, float]:
        """
        Update the user's rating for a topic after an interaction.

        Returns
        -------
        tuple[float, float, float]
            (rating_before, rating_after, delta)
        """
        tr = self.get_rating(topic)
        rating_before = tr.rating
        delta = tr.update(question_difficulty, answered_correctly, k_override)
        return rating_before, tr.rating, delta

    def summary(self) -> Dict[str, dict]:
        """Return a serialisable snapshot of all topic ratings."""
        return {
            topic: {"rating": round(tr.rating, 2), "matches": tr.matches}
            for topic, tr in self.ratings.items()
        }
