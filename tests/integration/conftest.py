"""Integration test fixtures for GPU-dependent tests.

This conftest uses a session-scoped fixture to ensure whisperx mock
is removed ONLY when integration tests actually run (not during collection
of unit tests).

Requirements:
- TESTING=0 (so lifespan loads real models)
- HF_TOKEN set (pyannote requires it)
- GPU available (CUDA)
"""

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def unmock_whisperx_for_integration():
    """Remove whisperx mock before any integration test imports.

    This is scoped to session and only applies to tests/integration/ directory.
    Unit tests in tests/ keep the mock via tests/conftest.py.
    """
    # CRITICAL: Remove whisperx mock BEFORE any app imports.
    sys.modules.pop("whisperx", None)
    sys.modules.pop("whisperx.diarize", None)

    # Disable TESTING flag so lifespan loads real models.
    os.environ.pop("TESTING", None)

    yield
