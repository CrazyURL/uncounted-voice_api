"""화자 재클러스터링 파이프라인 — Wave 4 옵션 B.

Phase 5: window builder (임베딩 추출 대상 window 생성).
Phase 6: AHC 재클러스터링 (다음 단계).
Phase 7: stt_processor 파이프라인 통합 (다음다음 단계).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Window 길이 기본값 (초). Phase 6/7에서 호출자가 오버라이드 가능.
_DEFAULT_MIN_WINDOW_SEC = 1.0
_DEFAULT_MAX_WINDOW_SEC = 4.0

# Gap tolerance: words within this gap are considered contiguous (seconds)
_SPEAKER_BOUNDARY_GAP_SEC = 0.5


@dataclass(frozen=True)
class EmbeddingWindow:
    """오디오 구간 중 화자 임베딩 추출 대상.

    Attributes:
        start: 절대 시간 (초). chunk-local 아님 — 청크 모드에서도 청크 시작 오프셋 적용 후 절대값.
        end: 절대 시간 (초).
        source: 입력 timeline 종류. "word" | "segment".
        word_indices: 이 window에 포함된 words 리스트 인덱스. immutable tuple.
    """
    start: float
    end: float
    source: str
    word_indices: tuple[int, ...]


def chunk_offset_to_absolute(time_in_chunk: float, chunk_start: float) -> float:
    """청크 로컬 시간을 절대 시간으로 변환.

    Args:
        time_in_chunk: 청크 내 로컬 시간 (초).
        chunk_start: 청크의 절대 시작 시간 (초).

    Returns:
        절대 시간 (초).
    """
    return chunk_start + time_in_chunk


def build_embedding_windows(
    words: list[dict],
    segments: list[dict] | None = None,
    *,
    min_window_seconds: float = _DEFAULT_MIN_WINDOW_SEC,
    max_window_seconds: float = _DEFAULT_MAX_WINDOW_SEC,
    audio_duration_sec: float | None = None,
    hop_on_speaker_boundaries: bool = True,
) -> tuple[EmbeddingWindow, ...]:
    """발화 word/segment timeline에서 임베딩 추출 대상 window 생성.

    - 화자 경계에서 끊고 (hop_on_speaker_boundaries=True 시 — Phase 5 기본)
    - 길이가 [min, max] 사이인 windows만 반환
    - audio_duration_sec이 주어지면 end를 그 안으로 clip
    - 입력 mutation 금지, 반환은 immutable tuple

    Args:
        words: Word records. Each record must have 'start', 'end', 'speaker' or 'speaker_id'.
        segments: Reserved for Phase 7 segment-level fallback. Unused in Phase 5.
        min_window_seconds: Minimum window duration (seconds). Shorter windows are skipped/merged.
        max_window_seconds: Maximum window duration (seconds). Longer runs are split.
        audio_duration_sec: If set, clip window.end to this value.
        hop_on_speaker_boundaries: If True, split windows at speaker changes (Phase 5 default).

    Returns:
        Tuple of EmbeddingWindow (immutable, in chronological order).
    """
    if not words:
        return ()

    # Extract speaker labels (prefer speaker_id, fall back to speaker)
    def get_speaker(word: dict) -> str:
        return word.get("speaker_id") or word.get("speaker") or "UNKNOWN"

    # Phase 1: Group words into contiguous same-speaker runs
    runs = []  # List of (start_idx, end_idx, speaker, time_range)
    current_run_start = 0
    current_speaker = get_speaker(words[0])

    for i in range(1, len(words)):
        word_speaker = get_speaker(words[i])
        prev_word_end = words[i - 1]["end"]
        curr_word_start = words[i]["start"]
        gap = curr_word_start - prev_word_end

        # Check if speaker changed or gap is too large
        if word_speaker != current_speaker or (hop_on_speaker_boundaries and gap > _SPEAKER_BOUNDARY_GAP_SEC):
            # End current run
            run_start_time = words[current_run_start]["start"]
            run_end_time = words[i - 1]["end"]
            runs.append((current_run_start, i - 1, current_speaker, run_start_time, run_end_time))

            # Start new run
            current_run_start = i
            current_speaker = word_speaker

    # Add final run
    run_start_time = words[current_run_start]["start"]
    run_end_time = words[-1]["end"]
    runs.append((current_run_start, len(words) - 1, current_speaker, run_start_time, run_end_time))

    # Phase 2: Split each run into max_window_seconds chunks, filter by min_window_seconds
    windows = []

    for run_start_idx, run_end_idx, speaker, run_start_time, run_end_time in runs:
        run_duration = run_end_time - run_start_time

        if run_duration < min_window_seconds:
            # Skip short runs (Phase 5 policy: skip instead of merge)
            continue

        # Split long runs into ≤ max_window_seconds chunks
        if run_duration <= max_window_seconds:
            # Single window for this run
            word_indices = tuple(range(run_start_idx, run_end_idx + 1))
            windows.append((run_start_time, run_end_time, "word", word_indices))
        else:
            # Split into multiple windows
            num_windows = int((run_duration + max_window_seconds - 1) // max_window_seconds)
            chunk_duration = run_duration / num_windows

            for chunk_i in range(num_windows):
                chunk_start = run_start_time + chunk_i * chunk_duration
                chunk_end = min(run_start_time + (chunk_i + 1) * chunk_duration, run_end_time)

                # Find words in this chunk
                chunk_words = []
                for idx in range(run_start_idx, run_end_idx + 1):
                    word = words[idx]
                    # Word overlaps with chunk if word.end > chunk_start and word.start < chunk_end
                    if word["end"] > chunk_start and word["start"] < chunk_end:
                        chunk_words.append(idx)

                if chunk_words:
                    word_indices = tuple(chunk_words)
                    windows.append((chunk_start, chunk_end, "word", word_indices))

    # Phase 3: Clip to audio bounds and filter final windows
    final_windows = []

    for start, end, source, word_indices in windows:
        # Clip to audio duration if specified
        if audio_duration_sec is not None:
            end = min(end, audio_duration_sec)
            if start >= end:
                # Window entirely outside bounds
                continue

        duration = end - start
        if duration >= min_window_seconds:
            final_windows.append(EmbeddingWindow(start, end, source, word_indices))

    return tuple(final_windows)
