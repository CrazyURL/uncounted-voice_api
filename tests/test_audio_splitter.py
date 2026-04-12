"""Tests for audio splitting utilities."""
import numpy as np
import pytest

from app.services.audio_splitter import (
    extract_speaker_audio,
    extract_utterance_audio,
    extract_utterance_audio_local,
    mute_non_speaker,
    to_wav_bytes,
)
from app.services.utterance_segmenter import UtteranceBoundary

SR = 16000


def _seg(start, end, speaker="SPEAKER_0"):
    return {"start": start, "end": end, "speaker": speaker}


class TestExtractSpeakerAudio:
    def test_extracts_single_speaker(self):
        audio = np.ones(SR * 10, dtype=np.float32)
        segments = [
            _seg(0.0, 3.0, "SPEAKER_0"),
            _seg(3.0, 6.0, "SPEAKER_1"),
            _seg(6.0, 10.0, "SPEAKER_0"),
        ]
        result = extract_speaker_audio(audio, segments, "SPEAKER_0", SR)
        assert result is not None
        expected_samples = int(3.0 * SR) + int(4.0 * SR)
        assert abs(len(result) - expected_samples) < 10

    def test_returns_none_for_short_duration(self):
        audio = np.ones(SR * 10, dtype=np.float32)
        segments = [_seg(0.0, 0.3, "SPEAKER_0")]
        result = extract_speaker_audio(audio, segments, "SPEAKER_0", SR)
        assert result is None

    def test_returns_none_for_missing_speaker(self):
        audio = np.ones(SR * 5, dtype=np.float32)
        segments = [_seg(0.0, 5.0, "SPEAKER_0")]
        result = extract_speaker_audio(audio, segments, "SPEAKER_99", SR)
        assert result is None


class TestExtractUtteranceAudio:
    def test_extracts_with_padding(self):
        audio = np.arange(SR * 10, dtype=np.float32)
        utt = UtteranceBoundary(
            start_sec=2.0, end_sec=5.0, duration_sec=3.0,
            padded_start_sec=1.85, padded_end_sec=5.15,
            speaker_id="SPEAKER_0", transcript_text="test", words=()
        )
        result = extract_utterance_audio(audio, utt, SR)
        expected = int(5.15 * SR) - int(1.85 * SR)
        assert abs(len(result) - expected) < 10


class TestExtractUtteranceAudioLocal:
    def test_slices_within_chunk(self):
        chunk = np.arange(SR * 10, dtype=np.float32)
        result = extract_utterance_audio_local(chunk, 2.0, 5.0, SR)
        assert len(result) == int(5.0 * SR) - int(2.0 * SR)
        assert result[0] == int(2.0 * SR)

    def test_clips_negative_start(self):
        chunk = np.arange(SR * 4, dtype=np.float32)
        result = extract_utterance_audio_local(chunk, -0.5, 1.0, SR)
        assert result[0] == 0
        assert len(result) == int(1.0 * SR)

    def test_clips_end_beyond_chunk(self):
        chunk = np.arange(SR * 3, dtype=np.float32)
        result = extract_utterance_audio_local(chunk, 1.0, 10.0, SR)
        assert len(result) == int(3.0 * SR) - int(1.0 * SR)

    def test_empty_slice_when_end_before_start(self):
        chunk = np.arange(SR * 5, dtype=np.float32)
        result = extract_utterance_audio_local(chunk, 3.0, 2.0, SR)
        assert len(result) == 0
        assert result.dtype == chunk.dtype

    def test_does_not_copy_array(self):
        chunk = np.ones(SR * 2, dtype=np.float32)
        result = extract_utterance_audio_local(chunk, 0.5, 1.5, SR)
        # Returned slice should be a view; mutating it changes the underlying chunk.
        result[0] = 42.0
        assert chunk[int(0.5 * SR)] == 42.0


class TestMuteNonSpeaker:
    def test_mutes_other_speakers(self):
        audio = np.ones(SR * 6, dtype=np.float32)
        segments = [
            _seg(0.0, 3.0, "SPEAKER_0"),
            _seg(3.0, 6.0, "SPEAKER_1"),
        ]
        result = mute_non_speaker(audio, segments, "SPEAKER_0", SR)
        assert np.all(result[int(3.0 * SR):int(6.0 * SR)] == 0.0)
        assert np.all(result[:int(3.0 * SR)] == 1.0)

    def test_does_not_mutate_original(self):
        audio = np.ones(SR * 4, dtype=np.float32)
        segments = [_seg(0.0, 2.0, "SPEAKER_1")]
        result = mute_non_speaker(audio, segments, "SPEAKER_0", SR)
        assert np.all(audio == 1.0)


class TestToWavBytes:
    def test_returns_valid_wav_bytes(self):
        audio = np.random.randn(SR * 2).astype(np.float32)
        result = to_wav_bytes(audio, SR)
        assert isinstance(result, bytes)
        assert len(result) > 44  # WAV header is 44 bytes minimum
        assert result[:4] == b"RIFF"  # WAV magic bytes
