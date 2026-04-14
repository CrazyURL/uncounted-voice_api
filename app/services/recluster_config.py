"""Recluster configuration contract (Phase 7: Option B WeSpeaker Integration).

Manages environment variables for WeSpeaker embedding extraction and reclustering.
All state is immutable (frozen dataclass).
"""

import os
from dataclasses import dataclass


class ReclusterConfigError(Exception):
    """Raised when recluster config is invalid."""

    pass


def _parse_bool(value: str) -> bool:
    """Parse string to boolean. Accepts true/false/1/0/yes/no (case-insensitive).

    Args:
        value: String value to parse.

    Returns:
        Boolean value.

    Raises:
        ReclusterConfigError: If value is not a recognized boolean string.
    """
    if not isinstance(value, str):
        raise ReclusterConfigError(f"Expected string, got {type(value).__name__}: {value}")

    normalized = value.strip().lower()

    if normalized in ("true", "1", "yes"):
        return True
    elif normalized in ("false", "0", "no"):
        return False
    else:
        raise ReclusterConfigError(
            f"Invalid boolean value: {value!r}. "
            "Expected one of: true, false, 1, 0, yes, no (case-insensitive)."
        )


def _parse_endpoints(value: str | None) -> frozenset[str]:
    """Parse comma-separated endpoint list.

    Args:
        value: Comma-separated string (e.g., "call_recording, voicemail") or None.

    Returns:
        Frozenset of endpoint names, stripped of whitespace.
        Default: frozenset(["call_recording"])
    """
    if not value:
        return frozenset(["call_recording"])

    endpoints = frozenset(ep.strip() for ep in value.split(",") if ep.strip())
    return endpoints if endpoints else frozenset(["call_recording"])


def _parse_float(value: str | None, default: float, param_name: str) -> float:
    """Parse string to float.

    Args:
        value: String value to parse or None.
        default: Default value if None.
        param_name: Parameter name for error message.

    Returns:
        Float value.

    Raises:
        ReclusterConfigError: If value is not a valid float.
    """
    if not value:
        return default

    try:
        return float(value.strip())
    except ValueError:
        raise ReclusterConfigError(
            f"Invalid float value for {param_name}: {value!r}. Expected a number."
        )


@dataclass(frozen=True)
class ReclusterConfig:
    """Immutable recluster configuration (Phase 7: Option B).

    Attributes:
        enabled: Enable WeSpeaker embedding extraction and reclustering.
        enabled_endpoint_modes: Frozenset of endpoint modes where recluster is active.
        confidence_threshold: Clustering confidence threshold (0.0-1.0). Below this, no relabeling.
        min_window_sec: Minimum window duration for embedding extraction (seconds).
        max_window_sec: Maximum window duration for embedding extraction (seconds).
    """

    enabled: bool
    enabled_endpoint_modes: frozenset[str]
    confidence_threshold: float
    min_window_sec: float
    max_window_sec: float

    @classmethod
    def from_env(cls) -> "ReclusterConfig":
        """Load config from environment variables.

        Environment variables:
            VOICE_DIARIZATION_WESPEAKER_RECLUSTER: true/false (default: false)
            VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS: comma-separated modes (default: call_recording)
            VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD: float 0.0-1.0 (default: 0.30)
            VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC: float (default: 0.5)
            VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC: float (default: 5.0)

        Returns:
            ReclusterConfig instance.

        Raises:
            ReclusterConfigError: If any env var value is invalid.
        """
        enabled_str = os.environ.get("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "false")
        enabled = _parse_bool(enabled_str)

        endpoints_str = os.environ.get(
            "VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS"
        )
        enabled_endpoint_modes = _parse_endpoints(endpoints_str)

        confidence_threshold = _parse_float(
            os.environ.get("VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD"),
            default=0.30,
            param_name="VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD",
        )

        min_window_sec = _parse_float(
            os.environ.get("VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC"),
            default=1.0,
            param_name="VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC",
        )

        max_window_sec = _parse_float(
            os.environ.get("VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC"),
            default=4.0,
            param_name="VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC",
        )

        if max_window_sec <= min_window_sec:
            raise ReclusterConfigError(
                f"max_window_sec ({max_window_sec}) must be > min_window_sec ({min_window_sec})"
            )

        return cls(
            enabled=enabled,
            enabled_endpoint_modes=enabled_endpoint_modes,
            confidence_threshold=confidence_threshold,
            min_window_sec=min_window_sec,
            max_window_sec=max_window_sec,
        )

    def is_enabled_for(self, mode: str) -> bool:
        """Check if recluster is enabled for a given endpoint mode.

        Args:
            mode: Endpoint mode (e.g., "call_recording", "voicemail").

        Returns:
            True if enabled globally AND mode is in allowlist, else False.
        """
        if not self.enabled:
            return False

        return mode in self.enabled_endpoint_modes
