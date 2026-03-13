"""
Unit tests for the Skill Engine — fully isolated, no DB, no API.

Validates the ELO formula properties mathematically:
  1. Correct answer always increases rating.
  2. Incorrect answer always decreases rating.
  3. Harder question yields larger gain on correct answer.
  4. K-factor schedule follows the defined tiers.
  5. Deterministic: same inputs → same outputs.
  6. Topic separation: updating one topic does not affect another.
"""

from __future__ import annotations

import math
import pytest

from app.skill_engine import (
    compute_new_rating,
    expected_score,
    get_k_factor,
    rating_delta,
    SkillProfile,
    TopicRating,
)


# ── K-factor schedule ───────────────────────────────────────────────────────

class TestKFactor:
    def test_new_user(self):
        """Fewer than 10 matches → K = 40."""
        for m in (0, 1, 5, 9):
            assert get_k_factor(m) == 40.0

    def test_intermediate_user(self):
        """10–29 matches → K = 32."""
        for m in (10, 15, 29):
            assert get_k_factor(m) == 32.0

    def test_established_user(self):
        """30+ matches → K = 24."""
        for m in (30, 50, 100):
            assert get_k_factor(m) == 24.0


# ── Expected score ──────────────────────────────────────────────────────────

class TestExpectedScore:
    def test_equal_ratings(self):
        """Equal R_u and R_q → expected = 0.5 exactly."""
        assert expected_score(1000.0, 1000.0) == pytest.approx(0.5)

    def test_strong_user(self):
        """User much stronger → expected close to 1."""
        e = expected_score(1400.0, 1000.0)
        assert e > 0.9

    def test_weak_user(self):
        """User much weaker → expected close to 0."""
        e = expected_score(600.0, 1000.0)
        assert e < 0.1

    def test_range(self):
        """Expected score is always in (0, 1)."""
        for r_u in (0, 500, 1000, 2000, 3000):
            for r_q in (0, 500, 1000, 2000, 3000):
                e = expected_score(float(r_u), float(r_q))
                assert 0.0 < e < 1.0

    def test_symmetry(self):
        """E(u, q) + E(q, u) == 1  (the two 'players' probabilities sum to 1)."""
        e1 = expected_score(1200.0, 1000.0)
        e2 = expected_score(1000.0, 1200.0)
        assert e1 + e2 == pytest.approx(1.0)


# ── Rating delta ────────────────────────────────────────────────────────────

class TestRatingDelta:
    def test_correct_answer_positive_delta(self):
        """Correct answer always produces a positive delta."""
        for r_u in (600, 800, 1000, 1200, 1400):
            for r_q in (600, 800, 1000, 1200, 1400):
                d = rating_delta(float(r_u), float(r_q), True, 32.0)
                assert d > 0.0, f"Failed at R_u={r_u}, R_q={r_q}"

    def test_incorrect_answer_negative_delta(self):
        """Incorrect answer always produces a negative delta."""
        for r_u in (600, 800, 1000, 1200, 1400):
            for r_q in (600, 800, 1000, 1200, 1400):
                d = rating_delta(float(r_u), float(r_q), False, 32.0)
                assert d < 0.0, f"Failed at R_u={r_u}, R_q={r_q}"

    def test_harder_question_larger_gain(self):
        """Correctly answering a harder question gives a bigger gain."""
        easy_delta = rating_delta(1000.0, 800.0, True, 32.0)
        hard_delta = rating_delta(1000.0, 1200.0, True, 32.0)
        assert hard_delta > easy_delta

    def test_easier_question_larger_loss(self):
        """Incorrectly answering an easier question gives a bigger loss."""
        easy_loss = rating_delta(1000.0, 800.0, False, 32.0)
        hard_loss = rating_delta(1000.0, 1200.0, False, 32.0)
        # easy_loss is more negative than hard_loss
        assert easy_loss < hard_loss


# ── compute_new_rating ──────────────────────────────────────────────────────

class TestComputeNewRating:
    def test_correct_increases_rating(self):
        new_r, delta = compute_new_rating(1000.0, 1000.0, True, 0)
        assert new_r > 1000.0
        assert delta > 0.0

    def test_incorrect_decreases_rating(self):
        new_r, delta = compute_new_rating(1000.0, 1000.0, False, 0)
        assert new_r < 1000.0
        assert delta < 0.0

    def test_k_override(self):
        """K-override bypasses the schedule."""
        _, delta_low_k = compute_new_rating(1000.0, 1000.0, True, 0, k_override=10.0)
        _, delta_high_k = compute_new_rating(1000.0, 1000.0, True, 0, k_override=50.0)
        assert abs(delta_high_k) > abs(delta_low_k)

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        r1 = compute_new_rating(1050.0, 980.0, True, 15)
        r2 = compute_new_rating(1050.0, 980.0, True, 15)
        assert r1 == r2

    def test_uses_k_schedule(self):
        """Verify the K-factor schedule is applied when no override."""
        # matches=5 → K=40, matches=25 → K=32
        _, delta_new = compute_new_rating(1000.0, 1000.0, True, 5)
        _, delta_mid = compute_new_rating(1000.0, 1000.0, True, 25)
        assert abs(delta_new) > abs(delta_mid)


# ── TopicRating ─────────────────────────────────────────────────────────────

class TestTopicRating:
    def test_default_values(self):
        tr = TopicRating()
        assert tr.rating == 1000.0
        assert tr.matches == 0

    def test_update_increments_matches(self):
        tr = TopicRating()
        tr.update(1000.0, True)
        assert tr.matches == 1
        tr.update(1000.0, False)
        assert tr.matches == 2

    def test_update_returns_delta(self):
        tr = TopicRating()
        delta = tr.update(1000.0, True)
        assert delta > 0.0

    def test_correct_moves_rating_up(self):
        tr = TopicRating()
        original = tr.rating
        tr.update(1000.0, True)
        assert tr.rating > original

    def test_incorrect_moves_rating_down(self):
        tr = TopicRating()
        original = tr.rating
        tr.update(1000.0, False)
        assert tr.rating < original

    def test_multiple_corrects_monotonic_increase(self):
        """Repeated correct answers keep pushing rating up."""
        tr = TopicRating()
        prev = tr.rating
        for _ in range(10):
            tr.update(1000.0, True)
            assert tr.rating > prev
            prev = tr.rating


# ── SkillProfile ────────────────────────────────────────────────────────────

class TestSkillProfile:
    def test_auto_creates_topic(self):
        sp = SkillProfile()
        tr = sp.get_rating("arrays")
        assert tr.rating == 1000.0
        assert tr.matches == 0

    def test_topic_isolation(self):
        """Updating one topic must not affect another."""
        sp = SkillProfile()
        sp.update("arrays", 1000.0, True)
        sp.update("arrays", 1000.0, True)
        sp.update("trees", 1000.0, False)

        assert sp.get_rating("arrays").rating > 1000.0
        assert sp.get_rating("trees").rating < 1000.0
        assert sp.get_rating("arrays").matches == 2
        assert sp.get_rating("trees").matches == 1

    def test_update_returns_triple(self):
        sp = SkillProfile()
        before, after, delta = sp.update("dp", 1000.0, True)
        assert before == 1000.0
        assert after > 1000.0
        assert delta == pytest.approx(after - before)

    def test_summary(self):
        sp = SkillProfile()
        sp.update("graphs", 1000.0, True)
        s = sp.summary()
        assert "graphs" in s
        assert "rating" in s["graphs"]
        assert "matches" in s["graphs"]
        assert s["graphs"]["matches"] == 1

    def test_unplayed_topic_not_in_summary(self):
        sp = SkillProfile()
        sp.update("arrays", 1000.0, True)
        s = sp.summary()
        assert "arrays" in s
        assert "trees" not in s


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_extreme_ratings(self):
        """Very high and very low ratings should not produce NaN or Inf."""
        new_r, delta = compute_new_rating(3000.0, 100.0, True, 50)
        assert math.isfinite(new_r)
        assert math.isfinite(delta)

        new_r, delta = compute_new_rating(100.0, 3000.0, False, 50)
        assert math.isfinite(new_r)
        assert math.isfinite(delta)

    def test_zero_rating(self):
        """Rating of 0 should work without errors."""
        new_r, delta = compute_new_rating(0.0, 1000.0, True, 0)
        assert math.isfinite(new_r)
        assert new_r > 0.0

    def test_negative_rating(self):
        """Negative ratings (possible after many losses) should still be computable."""
        new_r, delta = compute_new_rating(-200.0, 1000.0, False, 50)
        assert math.isfinite(new_r)
        assert new_r < -200.0

    def test_equal_difficulty_correct_gives_half_k(self):
        """When R_u == R_q and correct, delta should be K/2 (since E=0.5, S=1)."""
        _, delta = compute_new_rating(1000.0, 1000.0, True, 0, k_override=32.0)
        assert delta == pytest.approx(16.0)

    def test_equal_difficulty_incorrect_gives_neg_half_k(self):
        """When R_u == R_q and incorrect, delta should be -K/2."""
        _, delta = compute_new_rating(1000.0, 1000.0, False, 0, k_override=32.0)
        assert delta == pytest.approx(-16.0)
