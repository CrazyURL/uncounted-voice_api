"""Tests for diarization config contract (Phase 1: Option D).

Tests validate the DiarizationConfig class behavior:
- Boolean flag parsing (true/false/1/0/yes/no case-insensitive)
- Endpoint allowlist filtering
- Immutability (resolve_options returns new dict each call)
- Invalid env values fail at construction time
"""

import pytest


@pytest.mark.unit
class TestDiarizationConfigConstruction:
    """Tests for DiarizationConfig.from_env() initialization."""

    def test_force_two_speakers_default_is_false(self, monkeypatch):
        """Default VOICE_DIARIZATION_FORCE_TWO_SPEAKERS=false."""
        monkeypatch.delenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", raising=False)

        from app.services.diarization_config import DiarizationConfig

        config = DiarizationConfig.from_env()
        assert config.force_two_speakers is False

    def test_force_two_speakers_true_variants(self, monkeypatch):
        """Accepts true/1/yes (case-insensitive)."""
        from app.services.diarization_config import DiarizationConfig

        for value in ["true", "TRUE", "True", "1", "yes", "YES", "Yes"]:
            monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", value)
            config = DiarizationConfig.from_env()
            assert config.force_two_speakers is True, f"Failed for {value}"

    def test_force_two_speakers_false_variants(self, monkeypatch):
        """Accepts false/0/no (case-insensitive)."""
        from app.services.diarization_config import DiarizationConfig

        for value in ["false", "FALSE", "False", "0", "no", "NO", "No"]:
            monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", value)
            config = DiarizationConfig.from_env()
            assert config.force_two_speakers is False, f"Failed for {value}"

    def test_force_two_speakers_invalid_value_raises(self, monkeypatch):
        """Invalid value raises DiarizationConfigError at construction."""
        from app.services.diarization_config import DiarizationConfig, DiarizationConfigError

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "maybe")

        with pytest.raises(DiarizationConfigError) as exc_info:
            DiarizationConfig.from_env()
        assert "maybe" in str(exc_info.value).lower()

    def test_endpoints_allowlist_default(self, monkeypatch):
        """Default endpoints allowlist is call_recording."""
        monkeypatch.delenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", raising=False)

        from app.services.diarization_config import DiarizationConfig

        config = DiarizationConfig.from_env()
        assert "call_recording" in config.enabled_endpoint_modes

    def test_endpoints_allowlist_custom_comma_separated(self, monkeypatch):
        """Parses comma-separated endpoint list."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording, voicemail")

        config = DiarizationConfig.from_env()
        assert "call_recording" in config.enabled_endpoint_modes
        assert "voicemail" in config.enabled_endpoint_modes

    def test_enabled_endpoint_modes_is_frozenset(self, monkeypatch):
        """enabled_endpoint_modes is immutable."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.delenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", raising=False)
        config = DiarizationConfig.from_env()
        assert isinstance(config.enabled_endpoint_modes, frozenset)


@pytest.mark.unit
class TestResolveOptions:
    """Tests for resolve_options(mode) -> dict."""

    def test_call_recording_mode_with_flag_on_returns_two_speaker_hint(self, monkeypatch):
        """flag on + mode call_recording → {min_speakers: 2, max_speakers: 2}."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

        config = DiarizationConfig.from_env()
        options = config.resolve_options("call_recording")

        assert options == {"min_speakers": 2, "max_speakers": 2}

    def test_flag_off_returns_empty_dict(self, monkeypatch):
        """flag off + any mode → {}."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "false")
        config = DiarizationConfig.from_env()

        for mode in ["call_recording", "voicemail", "generic"]:
            options = config.resolve_options(mode)
            assert options == {}

    def test_non_call_endpoint_returns_empty_dict_when_not_in_allowlist(self, monkeypatch):
        """flag on + mode not in allowlist → {}."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

        config = DiarizationConfig.from_env()
        options = config.resolve_options("voicemail")

        assert options == {}

    def test_resolve_options_returns_new_dict_each_call(self, monkeypatch):
        """Each call returns a new dict (immutability proof)."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        config = DiarizationConfig.from_env()

        dict1 = config.resolve_options("call_recording")
        dict2 = config.resolve_options("call_recording")

        # Same content but different objects
        assert dict1 == dict2
        assert dict1 is not dict2

        # Mutating one doesn't affect the other
        dict1["extra_key"] = "extra_value"
        assert "extra_key" not in dict2

    def test_resolve_options_honors_custom_min_max_speakers(self, monkeypatch):
        """resolve_options uses min/max_speakers fields, not hardcoded."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
        config = DiarizationConfig.from_env()

        # Default config should use 2/2
        options = config.resolve_options("call_recording")
        assert options["min_speakers"] == config.min_speakers
        assert options["max_speakers"] == config.max_speakers


@pytest.mark.unit
class TestDataclassFrozen:
    """Tests for immutability (frozen dataclass)."""

    def test_config_is_frozen(self, monkeypatch):
        """Dataclass is frozen (immutable)."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.delenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", raising=False)
        config = DiarizationConfig.from_env()

        with pytest.raises(Exception):  # dataclass raises FrozenInstanceError or AttributeError
            config.force_two_speakers = True


@pytest.mark.unit
class TestDefaultEndpointAllowlist:
    """Tests for the default endpoint allowlist."""

    def test_default_endpoint_allowlist_contains_call_recording(self, monkeypatch):
        """Default allowlist must include 'call_recording'."""
        from app.services.diarization_config import DiarizationConfig

        monkeypatch.delenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", raising=False)
        config = DiarizationConfig.from_env()

        assert "call_recording" in config.enabled_endpoint_modes
