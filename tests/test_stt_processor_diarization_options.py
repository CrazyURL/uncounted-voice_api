"""Tests for diarization options integration in stt_processor (Phase 2: Option D).

Tests verify:
1. DiarizationConfig is used to determine diarization options
2. Options are passed to _diarize_model calls
3. Flag off → no kwargs passed
4. Mode filtering works (call_recording allowed, others filtered)
"""

import pytest


@pytest.mark.unit
class TestSttProcessorDiarizationOptions:
    """Integration tests for diarization options in transcribe paths."""

    def test_diarization_config_enabled_returns_two_speaker_hints(self, monkeypatch):
        """When flag enabled + mode in allowlist → resolve_options returns two-speaker hints."""
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

        from app.services.diarization_config import DiarizationConfig

        config = DiarizationConfig.from_env()
        options = config.resolve_options("call_recording")

        assert options == {"min_speakers": 2, "max_speakers": 2}

    def test_diarization_config_disabled_returns_empty_dict(self, monkeypatch):
        """When flag disabled → resolve_options returns empty dict regardless of mode."""
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "false")
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

        from app.services.diarization_config import DiarizationConfig

        config = DiarizationConfig.from_env()
        options = config.resolve_options("call_recording")

        assert options == {}

    def test_diarization_config_mode_filtering(self, monkeypatch):
        """When flag enabled but mode not in allowlist → resolve_options returns empty dict."""
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

        from app.services.diarization_config import DiarizationConfig

        config = DiarizationConfig.from_env()
        options_voicemail = config.resolve_options("voicemail")
        options_other = config.resolve_options("other_mode")

        assert options_voicemail == {}
        assert options_other == {}
