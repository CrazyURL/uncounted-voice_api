"""Regression tests for speaker diarization.

Task ID: 5dc13ea73c62 (utterance 431 speaker merge bug).
Tests use fixtures that reflect the current buggy state and expected reference labels.
"""

import pytest

from app.services.diarization_metrics import (
    count_speaker_label,
    speaker_count,
    word_level_speaker_accuracy,
)


@pytest.mark.unit
class TestDiarizationFixtures:
    """Fixture integrity tests."""

    def test_utterance_431_expected_fixture_contains_speaker_switch(
        self, utterance_431_expected
    ):
        """Verify that the expected fixture has at least one speaker boundary.

        This ensures our reference data is not degenerate (all one speaker).
        """
        assert len(utterance_431_expected) == 38
        labels = [word["speaker"] for word in utterance_431_expected]

        speaker_switches = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                speaker_switches += 1

        assert speaker_switches >= 1, "Expected fixture must contain at least one speaker boundary"


@pytest.mark.unit
class TestWordLevelMetrics:
    """Tests for word_level_speaker_accuracy metric."""

    def test_word_speaker_accuracy_detects_merged_failure(
        self, utterance_431_words, utterance_431_expected
    ):
        """Verify metric correctly flags the merged-speaker bug.

        All hypothesis words are SPEAKER_00 (merged), but reference has
        a speaker switch. Accuracy should be <0.80 to flag the problem.
        """
        accuracy = word_level_speaker_accuracy(
            utterance_431_words, utterance_431_expected
        )
        assert accuracy < 0.80, (
            f"Metric should detect merged speakers. Got accuracy={accuracy}"
        )

    def test_word_speaker_accuracy_perfect_match_returns_one(
        self, utterance_431_expected
    ):
        """Verify metric returns 1.0 when hypothesis equals reference."""
        accuracy = word_level_speaker_accuracy(
            utterance_431_expected, utterance_431_expected
        )
        assert accuracy == 1.0

    def test_word_speaker_accuracy_label_invariant(self, utterance_431_expected):
        """Verify metric is invariant to label swapping.

        Swapping SPEAKER_00 <-> SPEAKER_01 in the hypothesis should
        not change the permutation score.
        """
        swapped = []
        for word in utterance_431_expected:
            word_copy = dict(word)
            if word_copy["speaker"] == "SPEAKER_00":
                word_copy["speaker"] = "SPEAKER_01"
            elif word_copy["speaker"] == "SPEAKER_01":
                word_copy["speaker"] = "SPEAKER_00"
            swapped.append(word_copy)

        accuracy_original = word_level_speaker_accuracy(
            utterance_431_expected, utterance_431_expected
        )
        accuracy_swapped = word_level_speaker_accuracy(
            swapped, utterance_431_expected
        )

        assert accuracy_original == accuracy_swapped, (
            f"Label invariance failed: original={accuracy_original}, swapped={accuracy_swapped}"
        )


@pytest.mark.unit
class TestSpeakerCountMetrics:
    """Tests for speaker_count and count_speaker_label."""

    def test_speaker_count_handles_basic_cases(self):
        """Verify speaker_count counts distinct labels correctly."""
        assert speaker_count([]) == 0
        assert speaker_count(["SPEAKER_00"]) == 1
        assert speaker_count(["SPEAKER_00", "SPEAKER_01"]) == 2
        assert speaker_count(["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]) == 2

    def test_count_speaker_label_basic_cases(self):
        """Verify count_speaker_label counts exact matches."""
        assert count_speaker_label([], "SPEAKER_00") == 0
        assert count_speaker_label(["SPEAKER_00"], "SPEAKER_00") == 1
        assert count_speaker_label(["SPEAKER_00", "SPEAKER_01"], "SPEAKER_00") == 1
        assert (
            count_speaker_label(
                ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"], "SPEAKER_00"
            )
            == 2
        )

    def test_count_speaker_label_detects_phantom_speaker(
        self, utterance_431_words
    ):
        """Verify we can detect SPEAKER_02 (phantom speaker issue)."""
        words_with_phantom = utterance_431_words + [
            {"start": 3460.0, "end": 3460.5, "word": "phantom", "speaker": "SPEAKER_02"}
        ]
        count = count_speaker_label(
            [w["speaker"] for w in words_with_phantom], "SPEAKER_02"
        )
        assert count == 1
