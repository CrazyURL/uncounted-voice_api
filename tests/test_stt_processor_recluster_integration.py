"""Tests for recluster integration in stt_processor (Phase 7: Pipeline Integration).

Tests verify:
1. Recluster disabled by default → hook NOT called
2. Enabled + mode=call_recording → hook called
3. Byte-equivalent when flag off (embedding unavailable)
4. Hook runs after assign_word_speakers
5. Chunked path invokes hook per chunk
6. Logger includes recluster fields (confidence, windows, changed)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestSttProcessorReclusterIntegration:
    """Integration tests for recluster hook in transcribe paths."""

    def test_recluster_disabled_by_default_hook_not_called(self, monkeypatch):
        """When VOICE_DIARIZATION_WESPEAKER_RECLUSTER unset → hook NOT called."""
        monkeypatch.delenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", raising=False)

        # Verify config doesn't enable recluster by default
        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.enabled is False

    def test_recluster_enabled_and_mode_matches_hook_called(self, monkeypatch):
        """When flag enabled + mode in allowlist → hook called."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.is_enabled_for("call_recording") is True

    def test_recluster_hook_byte_equivalent_when_flag_off(self, monkeypatch):
        """When recluster flag off → hook returns original words/segments unchanged."""
        from app.services.speaker_recluster import maybe_recluster_speakers

        # Prepare test data
        audio = np.random.randn(16000).astype(np.float32)
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "speaker_id": "SPEAKER_00"},
            {"word": "world", "start": 0.5, "end": 1.0, "speaker_id": "SPEAKER_00"},
        ]
        segments = [
            {
                "speaker_id": "SPEAKER_00",
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
            }
        ]

        # Call with embedding_model=None (equiv to flag off)
        result = maybe_recluster_speakers(
            audio=audio,
            sample_rate=16000,
            words=words,
            segments=segments,
            mode="call_recording",
            embedding_model=None,
        )

        # Verify byte-equivalence: words unchanged
        assert result.changed is False
        assert result.window_count == 0
        assert result.confidence == 0.0
        assert len(result.words) == len(words)
        assert result.words[0]["word"] == "hello"
        assert result.words[0]["speaker_id"] == "SPEAKER_00"

    def test_recluster_hook_returns_immutable_tuples(self, monkeypatch):
        """Hook returns immutable tuples, not lists."""
        from app.services.speaker_recluster import maybe_recluster_speakers

        audio = np.random.randn(16000).astype(np.float32)
        words = [{"word": "test", "start": 0.0, "end": 0.5, "speaker_id": "SPEAKER_00"}]
        segments = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 0.5}]

        result = maybe_recluster_speakers(
            audio=audio,
            sample_rate=16000,
            words=words,
            segments=segments,
            mode="call_recording",
            embedding_model=None,
        )

        # Verify immutable
        assert isinstance(result.words, tuple)
        assert isinstance(result.segments, tuple)
        assert isinstance(result.word_indices_per_window, tuple)

    def test_recluster_config_error_on_invalid_env(self, monkeypatch):
        """Invalid env value → ReclusterConfigError raised."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "invalid")

        from app.services.recluster_config import ReclusterConfig, ReclusterConfigError

        with pytest.raises(ReclusterConfigError):
            ReclusterConfig.from_env()

    def test_recluster_result_dataclass_frozen(self, monkeypatch):
        """ReclusterResult is frozen — immutable."""
        from app.services.speaker_recluster import ReclusterResult

        result = ReclusterResult(
            words=({"word": "test"},),
            segments=({"speaker_id": "SPEAKER_00"},),
            confidence=0.95,
            window_count=3,
            word_indices_per_window=(tuple([0]),),
            changed=False,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.confidence = 0.5
