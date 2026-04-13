"""Integration tests for diarization long-call Option D gate (Phase 3).

These tests verify that Option D (min_speakers=2, max_speakers=2 hints)
correctly:
1. Suppresses phantom SPEAKER_02+ labels
2. Recovers speaker boundaries in problematic windows (utterance 431)
3. Does not regress processing time

Default fixture path (env var VOICE_API_LONG_CALL_FIXTURE):
  sample_data/Call/통화 녹음 임명훈_260313_195506.m4a
  (relative to workspace root /Users/gdash/project/uncounted-project/)

Markers: @pytest.mark.gpu, @pytest.mark.slow, @pytest.mark.real_audio
Skip conditions (ALL must be met to run):
  - RUN_GPU_INTEGRATION=1
  - VOICE_API_LONG_CALL_FIXTURE set and file exists
  - HF_TOKEN set (pyannote requirement)

Invocation strategy: Direct pipeline call via stt_processor.transcribe()
  (not TestClient) because we control TESTING=0 in conftest.py and
  only need result data, not HTTP response codes.

Artifact writing: Test outputs saved to tmp_path (pytest fixture),
  NOT to DB or /dev/shm (pytest teardown cleans up).
"""

import json
import os
import time
from pathlib import Path

import pytest
import soundfile


def _should_skip() -> str | None:
    """Check skip conditions. Return skip reason if any fail, else None."""
    if os.environ.get("RUN_GPU_INTEGRATION") != "1":
        return "RUN_GPU_INTEGRATION != '1'"

    fixture_path_env = os.environ.get("VOICE_API_LONG_CALL_FIXTURE")
    if not fixture_path_env:
        return "VOICE_API_LONG_CALL_FIXTURE not set"

    # Resolve relative to workspace root if relative path
    if not os.path.isabs(fixture_path_env):
        fixture_path = Path("/Users/gdash/project/uncounted-project") / fixture_path_env
    else:
        fixture_path = Path(fixture_path_env)

    if not fixture_path.exists():
        return f"VOICE_API_LONG_CALL_FIXTURE file not found: {fixture_path}"

    if not os.environ.get("HF_TOKEN"):
        return "HF_TOKEN not set (required for pyannote)"

    return None


skip_reason = _should_skip()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.real_audio,
    pytest.mark.skipif(skip_reason is not None, reason=skip_reason or ""),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.real_audio
def test_option_d_suppresses_phantom_speaker_02(monkeypatch, tmp_path):
    """Option D (two-speaker hint) should suppress SPEAKER_02 phantom labels.

    Asserts:
      - All word labels in result are in {SPEAKER_00, SPEAKER_01}
      - speaker_count <= 2
    """
    fixture_path_env = os.environ.get("VOICE_API_LONG_CALL_FIXTURE")
    if not os.path.isabs(fixture_path_env):
        fixture_path = Path("/Users/gdash/project/uncounted-project") / fixture_path_env
    else:
        fixture_path = Path(fixture_path_env)

    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

    from app.stt_processor import transcribe
    from app.services.diarization_metrics import speaker_count, count_speaker_label

    result = transcribe(
        file_path=str(fixture_path),
        task_id="test_phantom_speaker_02",
        enable_diarize=True,
        enable_name_masking=False,
        mask_pii=False,
        split_by_speaker=False,
        split_by_utterance=True,
        denoise_enabled=False,
    )

    # Extract word labels from result
    word_labels = []
    if "words" in result and result["words"]:
        word_labels = [w.get("speaker", "") for w in result["words"]]

    # All labels must be in {SPEAKER_00, SPEAKER_01}
    allowed_labels = {"SPEAKER_00", "SPEAKER_01"}
    for label in word_labels:
        assert label in allowed_labels, f"Found disallowed label: {label}"

    # Speaker count must be <= 2
    count = speaker_count(word_labels)
    assert count <= 2, f"speaker_count > 2: {count}"

    # Count SPEAKER_02 occurrences (should be 0)
    speaker_02_count = count_speaker_label(word_labels, "SPEAKER_02")
    assert speaker_02_count == 0, f"Found {speaker_02_count} SPEAKER_02 labels (expected 0)"

    # Save result for debugging
    result_path = tmp_path / "test_phantom_speaker_02_result.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "speaker_count": count,
                "speaker_02_count": speaker_02_count,
                "word_count": len(word_labels),
                "sample_labels": word_labels[:50] if word_labels else [],
            },
            f,
            indent=2,
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.real_audio
def test_option_d_recovers_utterance_431_boundary(monkeypatch, tmp_path):
    """Option D should recover the speaker boundary in utterance 431.

    Utterance 431 spans 3443.04~3472.63s. The problematic window is
    3449~3456s (7 seconds), which should contain a speaker boundary
    (SPEAKER_00 → other → SPEAKER_00).

    Asserts:
      - At least one speaker label boundary exists in [3449.0, 3456.0]
        (consecutive words with different speakers within window)
    """
    fixture_path_env = os.environ.get("VOICE_API_LONG_CALL_FIXTURE")
    if not os.path.isabs(fixture_path_env):
        fixture_path = Path("/Users/gdash/project/uncounted-project") / fixture_path_env
    else:
        fixture_path = Path(fixture_path_env)

    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

    from app.stt_processor import transcribe

    result = transcribe(
        file_path=str(fixture_path),
        task_id="test_utterance_431_boundary",
        enable_diarize=True,
        enable_name_masking=False,
        mask_pii=False,
        split_by_speaker=False,
        split_by_utterance=True,
        denoise_enabled=False,
    )

    # Find words in utterance 431 window
    window_start, window_end = 3449.0, 3456.0
    window_words = []
    if "words" in result and result["words"]:
        for word in result["words"]:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            # Include words whose start or end falls in the window
            if (word_start >= window_start and word_start <= window_end) or \
               (word_end >= window_start and word_end <= window_end):
                window_words.append(word)

    # Check for speaker boundary: consecutive words with different speakers
    boundary_found = False
    for i in range(len(window_words) - 1):
        curr_speaker = window_words[i].get("speaker", "")
        next_speaker = window_words[i + 1].get("speaker", "")
        if curr_speaker and next_speaker and curr_speaker != next_speaker:
            boundary_found = True
            break

    # Save detailed dump for debugging
    dump_path = tmp_path / "test_utterance_431_boundary_dump.json"
    with open(dump_path, "w") as f:
        json.dump(
            {
                "window_start": window_start,
                "window_end": window_end,
                "words_in_window": len(window_words),
                "boundary_found": boundary_found,
                "word_details": [
                    {
                        "start": w.get("start"),
                        "end": w.get("end"),
                        "speaker": w.get("speaker"),
                        "word": w.get("word"),
                    }
                    for w in window_words
                ],
            },
            f,
            indent=2,
        )

    assert boundary_found, (
        f"No speaker boundary found in [{window_start}, {window_end}]. "
        f"Found {len(window_words)} words in window. "
        f"See {dump_path} for details."
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.real_audio
def test_option_d_runtime_does_not_regress(monkeypatch):
    """Option D should not significantly regress processing time.

    Measures wall-clock time of full transcribe() call with flag on.
    Computes realtime multiplier = audio_duration / wall_clock.
    Asserts: multiplier >= 10.0x (baseline is ~16.1x on GPU server).

    Audio duration is read via soundfile.info().duration.
    """
    fixture_path_env = os.environ.get("VOICE_API_LONG_CALL_FIXTURE")
    if not os.path.isabs(fixture_path_env):
        fixture_path = Path("/Users/gdash/project/uncounted-project") / fixture_path_env
    else:
        fixture_path = Path(fixture_path_env)

    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "true")
    monkeypatch.setenv("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS_ENDPOINTS", "call_recording")

    from app.stt_processor import transcribe

    # Get audio duration
    audio_info = soundfile.info(str(fixture_path))
    audio_duration = audio_info.duration

    # Measure wall-clock time
    start_time = time.time()
    result = transcribe(
        file_path=str(fixture_path),
        task_id="test_runtime_regression",
        enable_diarize=True,
        enable_name_masking=False,
        mask_pii=False,
        split_by_speaker=False,
        split_by_utterance=True,
        denoise_enabled=False,
    )
    wall_clock = time.time() - start_time

    # Compute realtime multiplier
    rtf = audio_duration / wall_clock if wall_clock > 0 else 0

    # Soft lower bound: 10.0x (baseline is ~16.1x)
    assert rtf >= 10.0, (
        f"Realtime multiplier too low: {rtf:.1f}x "
        f"(audio {audio_duration:.1f}s, wall {wall_clock:.1f}s). "
        f"Expected >= 10.0x"
    )
