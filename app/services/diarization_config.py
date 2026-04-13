"""Diarization configuration contract (Phase 1: Option D).

Manages environment variables for two-speaker hints on specific endpoints.
All state is immutable (frozen dataclass).
"""

import os
from dataclasses import dataclass


class DiarizationConfigError(Exception):
    """Raised when diarization config is invalid."""
    pass


def _parse_bool(value: str) -> bool:
    """Parse string to boolean. Accepts true/false/1/0/yes/no (case-insensitive).

    Args:
        value: String value to parse.

    Returns:
        Boolean value.

    Raises:
        DiarizationConfigError: If value is not a recognized boolean string.
    """
    if not isinstance(value, str):
        raise DiarizationConfigError(f"Expected string, got {type(value).__name__}: {value}")

    normalized = value.strip().lower()

    if normalized in ("true", "1", "yes"):
        return True
    elif normalized in ("false", "0", "no"):
        return False
    else:
        raise DiarizationConfigError(
            f"Invalid boolean value: {value!r}. "
            "Expected one of: true, false, 1, 0, yes, no (case-insensitive)."
        )


def _parse_endpoints(value: str | None) -> frozenset[str]:
    """Parse comma-separated endpoint list.

    Args:
        value: Comma-separated string (e.g., "call_recording, voicemail") or None.

    Returns:
        Frozenset of endpoint names, stripped of whitespace.
    """
    if not value:
        return frozenset(["call_recording"])

    endpoints = frozenset(
        ep.strip() for ep in value.split(",") if ep.strip()
    )
    return endpoints if endpoints else frozenset(["call_recording"])


@dataclass(frozen=True)
class DiarizationConfig:
    """Immutable diarization configuration.

    Attributes:
        force_two_speakers: Enable two-speaker hints for specific endpoints.
        min_speakers: Minimum speakers hint (default 2 when enabled).
        max_speakers: Maximum speakers hint (default 2 when enabled).
        enabled_endpoint_modes: Frozenset of endpoint modes where hints are active.
    """

    force_two_speakers: bool
    min_speakers: int | None
    max_speakers: int | None
    enabled_endpoint_modes: frozenset[str]

    @classmethod
    def from_env(cls) -> "DiarizationConfig":
        """Load config from environment variables.

        Environment variables:
            VOICE_DIARIZATION_FORCE_TWO_SPEAKERS: true/false (default: false)
            VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS: comma-separated modes (default: call_recording)

        Returns:
            DiarizationConfig instance.

        Raises:
            DiarizationConfigError: If any env var value is invalid.
        """
        force_two_speakers_str = os.environ.get("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "false")
        force_two_speakers = _parse_bool(force_two_speakers_str)

        endpoints_str = os.environ.get("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS")
        enabled_endpoint_modes = _parse_endpoints(endpoints_str)

        # When enabled, use fixed min/max hints (2 speakers)
        min_speakers = 2 if force_two_speakers else None
        max_speakers = 2 if force_two_speakers else None

        return cls(
            force_two_speakers=force_two_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            enabled_endpoint_modes=enabled_endpoint_modes,
        )

    def resolve_options(self, mode: str) -> dict:
        """Resolve pyannote diarization options for a given endpoint mode.

        Returns a NEW dict on every call (immutability).

        Args:
            mode: Endpoint mode (e.g., "call_recording", "voicemail").

        Returns:
            Dict with min_speakers/max_speakers if enabled and mode is allowed, else empty dict.
        """
        if not self.force_two_speakers:
            return {}

        if mode not in self.enabled_endpoint_modes:
            return {}

        # Return a new dict each call (immutability proof)
        return {
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
        }
