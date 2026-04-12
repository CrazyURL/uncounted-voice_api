"""Tests for audio preprocessing pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.audio_preprocessor import (
    compress_silence,
    denoise,
    preprocess,
    remove_duplicates,
)

SR = 16000


def _sine(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    """Generate a sine wave at the given frequency and duration."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def _silence(duration: float, sr: int = SR) -> np.ndarray:
    """Generate silence of the given duration."""
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# TestDenoise
# ---------------------------------------------------------------------------

class TestDenoise:
    """Denoise는 DeepFilterNet 상주 subprocess 워커 기반이다.

    워커가 로드되지 않았거나 죽어있으면 입력을 그대로 반환한다.
    파일 IPC 경로(input.raw → request → output.raw)는 통합 테스트로 분리.
    """

    def test_returns_input_when_worker_not_ready(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        np.testing.assert_array_equal(result, audio)

    def test_returns_ndarray_same_dtype(self):
        audio = _sine(440, 1.0).astype(np.float32)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_does_not_mutate_input(self):
        audio = _sine(440, 1.0)
        original = audio.copy()
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            denoise(audio, SR)
        np.testing.assert_array_equal(audio, original)

    def test_skips_when_worker_none(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._df_process", new=None):
            result = denoise(audio, SR)
        np.testing.assert_array_equal(result, audio)

    def test_output_is_1d(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert result.ndim == 1
        assert result.shape == audio.shape

    def test_handles_silence_input(self):
        audio = _silence(1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert len(result) == len(audio)


# ---------------------------------------------------------------------------
# TestRemoveDuplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_repeated_segment(self):
        # Arrange — same 3-second segment repeated twice
        segment = _sine(440, 3.0)
        audio = np.concatenate([segment, segment])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — should be shorter (one copy removed)
        assert len(result) < len(audio)
        assert len(result) >= len(segment)

    def test_keeps_unique_segments(self):
        # Arrange — two different frequencies
        seg_a = _sine(440, 3.0)
        seg_b = _sine(880, 3.0)
        audio = np.concatenate([seg_a, seg_b])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — no removal, length preserved
        assert len(result) == len(audio)

    def test_skips_silent_regions(self):
        # Arrange — two silence blocks (should NOT be flagged as duplicates)
        audio = np.concatenate([_silence(3.0), _sine(440, 1.0), _silence(3.0)])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — no removal
        assert len(result) == len(audio)

    def test_does_not_mutate_input(self):
        # Arrange
        segment = _sine(440, 3.0)
        audio = np.concatenate([segment, segment])
        original = audio.copy()

        # Act
        remove_duplicates(audio, SR)

        # Assert
        np.testing.assert_array_equal(audio, original)

    def test_handles_short_audio(self):
        # Arrange — shorter than one window
        audio = _sine(440, 1.0)

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — returned as-is
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# TestCompressSilence
# ---------------------------------------------------------------------------

class TestCompressSilence:
    def test_compresses_long_silence(self):
        # Arrange — speech + 3s silence + speech
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — 3s silence compressed to ~0.5s
        expected_max = len(speech) * 2 + int(0.5 * SR) + SR  # some tolerance
        assert len(result) < len(audio)
        assert len(result) < expected_max

    def test_preserves_short_silence(self):
        # Arrange — speech + 0.3s silence + speech (below threshold)
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(0.3)
        audio = np.concatenate([speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — no compression, length preserved
        assert len(result) == len(audio)

    def test_does_not_mutate_input(self):
        # Arrange
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])
        original = audio.copy()

        # Act
        compress_silence(audio, SR)

        # Assert
        np.testing.assert_array_equal(audio, original)

    def test_output_shorter_with_gaps(self):
        # Arrange — multiple long silence gaps
        speech = _sine(440, 0.5) * 0.5
        silence = _silence(2.0)
        audio = np.concatenate([speech, silence, speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — both gaps compressed
        assert len(result) < len(audio)
        # Original: 0.5*3 + 2.0*2 = 5.5s → compressed: 0.5*3 + 0.5*2 = 2.5s
        expected_approx = int(2.5 * SR)
        assert abs(len(result) - expected_approx) < SR  # within 1s tolerance


# ---------------------------------------------------------------------------
# TestPreprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    """Preprocess 파이프라인은 config flag 기반으로 각 단계가 독립 제어된다."""

    @staticmethod
    def _enable_all(monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", True)

    @staticmethod
    def _disable_all(monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", False)

    @patch("app.services.audio_preprocessor.denoise")
    def test_chains_all_steps_when_all_enabled(self, mock_denoise, monkeypatch):
        # Arrange — all flags on, denoise mocked to pass through
        self._enable_all(monkeypatch)
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(audio, SR)

        # Assert — denoise called, silence compressed
        mock_denoise.assert_called_once()
        assert len(result) < len(audio)

    @patch("app.services.audio_preprocessor.denoise")
    def test_returns_numpy_float32(self, mock_denoise, monkeypatch):
        # Arrange
        self._enable_all(monkeypatch)
        audio = _sine(440, 2.0).astype(np.float32)
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(audio, SR)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @patch("app.services.audio_preprocessor.denoise")
    def test_all_flags_off_returns_unchanged_length(self, mock_denoise, monkeypatch):
        # Arrange — all flags off
        self._disable_all(monkeypatch)
        audio = _sine(440, 2.0).astype(np.float32) * 0.5
        silence = _silence(3.0)
        full = np.concatenate([audio, silence, audio])
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(full, SR)

        # Assert — denoise never called, length preserved
        mock_denoise.assert_not_called()
        assert len(result) == len(full)

    @patch("app.services.audio_preprocessor.denoise")
    def test_only_gain_enabled_round_1(self, mock_denoise, monkeypatch):
        # Arrange — Round 1 default: only gain
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", False)

        # Low-amplitude audio so gain actually applies
        audio = (_sine(440, 2.0) * 0.02).astype(np.float32)
        silence = _silence(3.0).astype(np.float32)
        full = np.concatenate([audio, silence, audio])
        original_len = len(full)

        # Act
        result = preprocess(full, SR)

        # Assert — denoise not called, length preserved (no silence compress),
        # gain boosted amplitude
        mock_denoise.assert_not_called()
        assert len(result) == original_len
        orig_rms = float(np.sqrt(np.mean(full ** 2)))
        new_rms = float(np.sqrt(np.mean(result ** 2)))
        assert new_rms > orig_rms  # gain amplified

    @patch("app.services.audio_preprocessor.denoise")
    def test_denoise_flag_controls_invocation(self, mock_denoise, monkeypatch):
        # Arrange — denoise off, others off
        self._disable_all(monkeypatch)
        audio = _sine(440, 1.0).astype(np.float32)
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        preprocess(audio, SR)

        # Assert
        mock_denoise.assert_not_called()

        # Flip denoise on → should be called
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", True)
        preprocess(audio, SR)
        mock_denoise.assert_called_once()
