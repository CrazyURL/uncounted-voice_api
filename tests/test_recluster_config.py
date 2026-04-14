"""Tests for ReclusterConfig (Phase 7: Configuration).

Tests verify:
1. ReclusterConfig.from_env() parses environment variables correctly
2. is_enabled_for(mode) filters by allowlist
3. Flag disabled → False regardless of mode
4. Invalid env value → ReclusterConfigError (fail-closed)
5. Immutable frozen dataclass
6. Defaults match interface-freeze spec
"""

import pytest


@pytest.mark.unit
class TestReclusterConfig:
    """Unit tests for ReclusterConfig frozen dataclass."""

    def test_recluster_config_enabled_returns_true_for_allowed_mode(self, monkeypatch):
        """When flag enabled + mode in allowlist → is_enabled_for returns True."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.is_enabled_for("call_recording") is True

    def test_recluster_config_disabled_returns_false(self, monkeypatch):
        """When flag disabled → is_enabled_for returns False regardless of mode."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "false")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.is_enabled_for("call_recording") is False
        assert config.is_enabled_for("voicemail") is False
        assert config.is_enabled_for("other") is False

    def test_recluster_config_mode_filtering(self, monkeypatch):
        """When flag enabled but mode not in allowlist → is_enabled_for returns False."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.is_enabled_for("call_recording") is True
        assert config.is_enabled_for("voicemail") is False
        assert config.is_enabled_for("other_mode") is False

    def test_recluster_config_invalid_bool_raises_error(self, monkeypatch):
        """When VOICE_DIARIZATION_WESPEAKER_RECLUSTER is invalid → ReclusterConfigError."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "maybe")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig, ReclusterConfigError

        with pytest.raises(ReclusterConfigError):
            ReclusterConfig.from_env()

    def test_recluster_config_frozen_immutable(self, monkeypatch):
        """ReclusterConfig is frozen — cannot mutate after creation."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", "call_recording")

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.enabled = False

    def test_recluster_config_defaults_match_spec(self, monkeypatch):
        """ReclusterConfig defaults match interface-freeze.md spec."""
        monkeypatch.delenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", raising=False)
        monkeypatch.delenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS", raising=False)
        monkeypatch.delenv("VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC", raising=False)
        monkeypatch.delenv("VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC", raising=False)

        from app.services.recluster_config import ReclusterConfig

        config = ReclusterConfig.from_env()
        assert config.enabled is False  # default: "false"
        assert "call_recording" in config.enabled_endpoint_modes  # default includes call_recording
        assert config.confidence_threshold > 0  # TBD but > 0
        assert config.min_window_sec > 0
        assert config.max_window_sec > config.min_window_sec
