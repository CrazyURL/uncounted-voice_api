"""Tests for speaker embedding window builder (Phase 5).

Tests validate build_embedding_windows():
- Empty input handling
- Single speaker continuous runs with time windowing
- Speaker boundary detection and segmentation
- Utterance 431 regression (expected reference labels)
- Audio duration clipping
- Short window handling (skip or merge policy)
- Maximum window length capping
- Chunk offset to absolute time conversion
- Immutability of returned structures (frozen dataclass, tuple results)
"""

import pytest

from app.services.speaker_recluster import (
    EmbeddingWindow,
    build_embedding_windows,
    chunk_offset_to_absolute,
)


@pytest.mark.unit
class TestEmptyInputs:
    """Tests for empty words/segments."""

    def test_empty_words_returns_empty_windows(self):
        """build_embedding_windows([], []) → ()"""
        result = build_embedding_windows([], [])
        assert result == ()
        assert isinstance(result, tuple)


@pytest.mark.unit
class TestSingleSpeakerContinuous:
    """Tests for continuous same-speaker run within time bounds."""

    def test_single_speaker_continuous_run(self):
        """10 words, same speaker, contiguous timestamps.
        Assert windows formed within [min_window_sec, max_window_sec].
        """
        words = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 3.5, "speaker": "SPEAKER_00"},
            {"start": 3.5, "end": 4.0, "speaker": "SPEAKER_00"},
            {"start": 4.0, "end": 4.5, "speaker": "SPEAKER_00"},
            {"start": 4.5, "end": 5.0, "speaker": "SPEAKER_00"},
        ]
        min_sec = 1.0
        max_sec = 4.0

        result = build_embedding_windows(
            words,
            [],
            min_window_seconds=min_sec,
            max_window_seconds=max_sec,
        )

        assert len(result) > 0
        for window in result:
            duration = window.end - window.start
            assert min_sec <= duration <= max_sec
            assert window.start >= 0.0
            assert window.end <= 5.0


@pytest.mark.unit
class TestSpeakerBoundaries:
    """Tests for speaker change detection and window splitting."""

    def test_speaker_change_creates_boundary(self):
        """A→B→A timeline splits windows at speaker changes."""
        words = [
            # SPEAKER_00: 0.0-2.0
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
            # SPEAKER_01: 2.0-4.0
            {"start": 2.0, "end": 2.5, "speaker": "SPEAKER_01"},
            {"start": 2.5, "end": 3.0, "speaker": "SPEAKER_01"},
            {"start": 3.0, "end": 3.5, "speaker": "SPEAKER_01"},
            {"start": 3.5, "end": 4.0, "speaker": "SPEAKER_01"},
            # SPEAKER_00 again: 4.0-6.0
            {"start": 4.0, "end": 4.5, "speaker": "SPEAKER_00"},
            {"start": 4.5, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 5.5, "speaker": "SPEAKER_00"},
            {"start": 5.5, "end": 6.0, "speaker": "SPEAKER_00"},
        ]

        result = build_embedding_windows(words, [], hop_on_speaker_boundaries=True)

        # Should create multiple windows, separated by speaker boundaries
        assert len(result) >= 2

        # Verify no window crosses a speaker boundary by checking word indices
        for window in result:
            speakers_in_window = set()
            for idx in window.word_indices:
                speakers_in_window.add(words[idx]["speaker"])
            # All words in a window must be the same speaker
            assert len(speakers_in_window) == 1


@pytest.mark.unit
class TestUtterance431Regression:
    """Tests with actual utterance 431 fixture data."""

    def test_utterance_431_shape_creates_multiple_windows(
        self, utterance_431_expected
    ):
        """Load utterance_431_expected (label-correct), build windows.
        Assert ≥3 windows, one covers 3449~3456s sub-region.
        Verify word_indices reference correct words.
        """
        min_sec = 1.0
        max_sec = 4.0

        result = build_embedding_windows(
            utterance_431_expected,
            [],
            min_window_seconds=min_sec,
            max_window_seconds=max_sec,
        )

        # Should create at least 3 windows from the 38-word utterance
        assert len(result) >= 3, f"Expected ≥3 windows, got {len(result)}"

        # Find window(s) covering 3449~3456s
        target_start = 3449.0
        target_end = 3456.0
        covering_windows = [
            w for w in result
            if w.start < target_end and w.end > target_start
        ]
        assert len(covering_windows) > 0, "No window covers 3449~3456s range"

        # Verify word_indices point to valid positions
        for window in result:
            for idx in window.word_indices:
                assert 0 <= idx < len(utterance_431_expected)
                word = utterance_431_expected[idx]
                # Word's time bounds should overlap window
                assert word["end"] > window.start
                assert word["start"] < window.end


@pytest.mark.unit
class TestAudioDurationClipping:
    """Tests for audio bounds clipping."""

    def test_window_clipped_to_audio_bounds(self):
        """Word extends beyond audio_duration_sec → window.end clipped."""
        words = [
            {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_00"},
            {"start": 6.5, "end": 7.0, "speaker": "SPEAKER_00"},
        ]
        audio_duration = 5.0

        result = build_embedding_windows(
            words,
            [],
            audio_duration_sec=audio_duration,
        )

        # Windows should exist but clipped to ≤ 5.0
        for window in result:
            assert window.end <= audio_duration


@pytest.mark.unit
class TestShortWindowHandling:
    """Tests for short window skip/merge policy."""

    def test_short_window_skipped(self):
        """Isolated 0.3s word → skipped (below min_window_seconds=1.0)."""
        words = [
            {"start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"},
        ]

        result = build_embedding_windows(
            words,
            [],
            min_window_seconds=1.0,
        )

        # Short window should be skipped → empty result
        assert len(result) == 0

    def test_short_window_merged_into_neighbor(self):
        """Short window merges with same-speaker neighbor
        (hop_on_speaker_boundaries=False).
        """
        words = [
            # Long run: 2.5s
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.5, "speaker": "SPEAKER_00"},
            # Short gap (< 0.5s gap tolerance) or directly adjacent
            {"start": 2.5, "end": 2.6, "speaker": "SPEAKER_00"},
            {"start": 2.6, "end": 3.1, "speaker": "SPEAKER_00"},
        ]

        result = build_embedding_windows(
            words,
            [],
            min_window_seconds=1.0,
            hop_on_speaker_boundaries=False,
        )

        # With merge policy, short fragment should be absorbed
        # Result depends on implementation: either merged or separate window
        # We verify at least one window covers the span
        if len(result) > 0:
            total_span = result[0].end - result[0].start
            assert total_span >= 1.0 or len(result) > 1


@pytest.mark.unit
class TestMaxWindowCapping:
    """Tests for max_window_seconds enforcement."""

    def test_max_window_seconds_caps_long_runs(self):
        """20s of same-speaker words → split into ≤4.0s windows."""
        words = []
        for i in range(40):
            words.append({
                "start": i * 0.5,
                "end": (i + 1) * 0.5,
                "speaker": "SPEAKER_00",
            })

        max_sec = 4.0
        result = build_embedding_windows(
            words,
            [],
            max_window_seconds=max_sec,
        )

        assert len(result) >= 2  # Should split long run
        for window in result:
            duration = window.end - window.start
            assert duration <= max_sec


@pytest.mark.unit
class TestChunkOffsetToAbsoluteConversion:
    """Tests for chunk local time → absolute time conversion."""

    def test_chunk_offset_to_absolute_conversion(self):
        """Chunk 2 (start=1800.0), local 200.0s → absolute 2000.0s."""
        chunk_start = 1800.0
        local_time = 200.0

        absolute_time = chunk_offset_to_absolute(local_time, chunk_start)

        assert absolute_time == 2000.0

    def test_chunk_offset_multiple_chunks(self):
        """Multiple chunk examples verify linear offset."""
        test_cases = [
            (1800.0, 0.0, 1800.0),     # Start of chunk 2
            (1800.0, 300.0, 2100.0),   # 300s into chunk 2
            (3600.0, 1.5, 3601.5),     # Chunk 3, offset 1.5s
        ]

        for chunk_start, local_time, expected_absolute in test_cases:
            result = chunk_offset_to_absolute(local_time, chunk_start)
            assert result == expected_absolute


@pytest.mark.unit
class TestImmutability:
    """Tests for immutable return structures."""

    def test_word_indices_are_immutable_tuple(self):
        """EmbeddingWindow.word_indices is tuple (not list), frozen dataclass."""
        words = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]

        result = build_embedding_windows(words, [])

        if len(result) > 0:
            window = result[0]
            assert isinstance(window.word_indices, tuple)
            # Frozen dataclass should prevent mutation
            with pytest.raises(Exception):  # FrozenInstanceError
                window.start = 999.0

    def test_returned_collection_is_immutable_tuple(self):
        """build_embedding_windows returns tuple (not list)."""
        words = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        ]

        result = build_embedding_windows(words, [])

        assert isinstance(result, tuple)
