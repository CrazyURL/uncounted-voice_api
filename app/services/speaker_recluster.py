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


# ── Phase 6 — recluster ──────────────────────────────────────────────────

import numpy as np
from sklearn.cluster import AgglomerativeClustering

_DEFAULT_CONFIDENCE_THRESHOLD = 0.15


def recluster_speakers(
    words: list[dict],
    embeddings: np.ndarray,
    window_indices_per_word: list[int | None],
    *,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[tuple[dict, ...], float, bool]:
    """2-speaker AHC 재클러스터링 + 라벨 재매핑.

    Takes the original word list, per-window embeddings, and a mapping from
    word index to window index. Performs 2-cluster agglomerative hierarchical
    clustering on the embeddings using cosine distance and average linkage.
    Computes a confidence score (intra-cluster vs inter-cluster margin).
    If confidence < threshold, returns original labels unchanged.
    Otherwise, relabels each word based on its window's cluster assignment.

    Args:
        words: List of word dicts. Input mutation is forbidden.
        embeddings: shape (W, D), L2-normalized. W = number of windows.
        window_indices_per_word: parallel to words; None means no embedding for that word.
        confidence_threshold: If confidence < threshold, return unchanged.

    Returns:
        (updated_words, confidence, changed)
        - updated_words: tuple of new word dicts (copies, never mutating input)
        - confidence: [0.0, 1.0] confidence score
        - changed: True if any word label changed
    """
    if not words or embeddings.shape[0] == 0:
        return (), 0.0, False

    # Compute 2-cluster AHC on embeddings
    cluster_labels = _cluster_two(embeddings)

    # Compute confidence score
    confidence = _compute_confidence(embeddings, cluster_labels)

    # If confidence below threshold, return unchanged
    if confidence < confidence_threshold:
        immutable_words = tuple(dict(w) for w in words)
        return immutable_words, confidence, False

    # Build mapping: window index → cluster label (0 or 1)
    window_to_cluster = {}
    for window_idx, cluster_label in enumerate(cluster_labels):
        window_to_cluster[window_idx] = cluster_label

    # Canonicalize cluster ids to speaker labels
    canonical_labels = _canonicalize_labels(cluster_labels)

    # Relabel words based on window cluster assignments
    changed = False
    updated_words_list = []

    for i, word in enumerate(words):
        updated_word = dict(word)
        window_idx = window_indices_per_word[i]

        # If word is mapped to a window, apply cluster-based label
        if window_idx is not None and window_idx in window_to_cluster:
            cluster_id = window_to_cluster[window_idx]
            new_label = canonical_labels[cluster_id]
            if updated_word.get("speaker_id") != new_label:
                updated_word["speaker_id"] = new_label
                changed = True
        # If unmapped, keep original label (no change)

        updated_words_list.append(updated_word)

    return tuple(updated_words_list), confidence, changed


def _cluster_two(embeddings: np.ndarray) -> np.ndarray:
    """Pure-numpy AHC: 2-cluster agglomerative hierarchical clustering.

    Uses cosine distance and average linkage. Returns cluster labels (0 or 1).

    Args:
        embeddings: shape (N, D), L2-normalized.

    Returns:
        shape (N,), dtype int. Cluster labels (0 or 1).
    """
    n = embeddings.shape[0]

    # Edge cases
    if n <= 1:
        # Single or no embedding → all cluster 0
        return np.zeros(n, dtype=int)

    if n == 2:
        # Two embeddings → one per cluster
        return np.array([0, 1], dtype=int)

    # Compute pairwise cosine distance matrix
    # cosine_similarity = dot product (since L2-normalized)
    # cosine_distance = 1 - cosine_similarity
    similarity = embeddings @ embeddings.T
    distance = 1.0 - similarity

    # Initialize: each embedding is its own cluster
    # cluster_members[i] = list of original indices in cluster i
    cluster_members = [[i] for i in range(n)]

    # Agglomerative clustering: repeatedly merge nearest pair until 2 clusters remain
    while len(cluster_members) > 2:
        # Find pair of clusters with minimum average distance
        min_dist = float('inf')
        merge_i, merge_j = 0, 1

        for i in range(len(cluster_members)):
            for j in range(i + 1, len(cluster_members)):
                # Average distance between cluster i and j
                cluster_i_indices = cluster_members[i]
                cluster_j_indices = cluster_members[j]

                avg_dist = 0.0
                for idx_i in cluster_i_indices:
                    for idx_j in cluster_j_indices:
                        avg_dist += distance[idx_i, idx_j]
                avg_dist /= (len(cluster_i_indices) * len(cluster_j_indices))

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        # Merge cluster merge_j into cluster merge_i
        cluster_members[merge_i].extend(cluster_members[merge_j])
        cluster_members.pop(merge_j)

    # Assign labels: cluster 0 and cluster 1
    labels = np.zeros(n, dtype=int)
    for cluster_id, indices_in_cluster in enumerate(cluster_members):
        for idx in indices_in_cluster:
            labels[idx] = cluster_id

    return labels


def _compute_confidence(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute confidence score: intra vs inter-cluster margin.

    margin = mean(intra_cluster_cosine_sim) - mean(inter_cluster_cosine_sim)
    confidence = clip(margin, [0, 1])

    Args:
        embeddings: shape (N, D), L2-normalized.
        labels: shape (N,), cluster assignments (0 or 1).

    Returns:
        float in [0.0, 1.0].
    """
    # Compute cosine similarities
    similarity = embeddings @ embeddings.T

    intra_sims = []
    inter_sims = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = similarity[i, j]
            if labels[i] == labels[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    # Compute means
    intra_mean = np.mean(intra_sims) if intra_sims else 1.0
    inter_mean = np.mean(inter_sims) if inter_sims else 0.0

    margin = intra_mean - inter_mean

    # Clip to [0, 1]
    confidence = max(0.0, min(1.0, margin))

    return float(confidence)


def _canonicalize_labels(cluster_assignments: np.ndarray) -> list[str]:
    """Map cluster ids (0, 1) to canonical labels (SPEAKER_00, SPEAKER_01).

    Deterministic: lower cluster id → SPEAKER_00. If a cluster has no first
    occurrence or to ensure determinism, swap clusters so cluster 0 occurs first.

    Args:
        cluster_assignments: shape (N,), values 0 or 1.

    Returns:
        list of str: [label_for_cluster_0, label_for_cluster_1]
    """
    # Find first occurrence of each cluster
    first_occurrence = {}
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in first_occurrence:
            first_occurrence[cluster_id] = i

    # Sort clusters by first occurrence to ensure determinism
    sorted_clusters = sorted(first_occurrence.keys(), key=lambda c: first_occurrence[c])

    # Assign labels: first cluster → SPEAKER_00, second → SPEAKER_01
    canonical = [""] * 2
    for position, cluster_id in enumerate(sorted_clusters):
        if position == 0:
            canonical[cluster_id] = "SPEAKER_00"
        elif position == 1:
            canonical[cluster_id] = "SPEAKER_01"

    return canonical


# ── Phase 7 — Pipeline Integration ──────────────────────────────────────────

from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable


@dataclass(frozen=True)
class ReclusterResult:
    """Result of maybe_recluster_speakers (Phase 7 integration hook).

    Immutable frozen dataclass. All outputs are tuples (no list references).

    Attributes:
        words: tuple of word dicts (immutable copy). If changed=False or flag off,
               identical content to input but as a tuple.
        segments: tuple of segment dicts (immutable copy). Typically unchanged from input.
        confidence: float [0.0, 1.0]. Clustering confidence. 0.0 if flag off or unavailable.
        window_count: int. Number of embedding windows that were built. 0 if flag off.
        word_indices_per_window: tuple of tuples. For each window, the word indices it contains.
                                 Empty if flag off.
        changed: bool. True if at least one word.speaker_id was modified.
    """

    words: tuple[dict, ...]
    segments: tuple[dict, ...]
    confidence: float
    window_count: int
    word_indices_per_window: tuple[tuple[int, ...], ...]
    changed: bool


def maybe_recluster_speakers(
    audio: np.ndarray,
    sample_rate: int,
    words: list[dict],
    segments: list[dict],
    mode: str,
    embedding_model: SpeakerEmbeddingModel | None = None,
) -> ReclusterResult:
    """Optional WeSpeaker reclustering hook (Phase 7: Option B Integration).

    If embedding_model is None or unavailable, returns immutable copy of inputs
    with changed=False and window_count=0 (byte-equivalent behavior).

    Otherwise:
    1. Build embedding windows from words/segments
    2. Extract embeddings from windows
    3. Perform 2-cluster AHC reclustering
    4. Relabel words if confidence >= threshold
    5. Return immutable ReclusterResult

    Caller must configure ReclusterConfig separately (enable flag, endpoints,
    confidence threshold, window size limits).

    Args:
        audio: mono float32 array, shape (N,). 16 kHz expected.
        sample_rate: sample rate (Hz). Typically 16000.
        words: word dicts with 'start', 'end', 'speaker_id'. NOT mutated.
        segments: segment dicts. NOT mutated.
        mode: endpoint mode (e.g., "call_recording") for logging/filtering.
        embedding_model: SpeakerEmbeddingModel or None. If None, returns unchanged.

    Returns:
        ReclusterResult (frozen, immutable). Input dicts are never mutated.
    """
    # Phase 7a: Bypass if model unavailable or flag not set by caller
    if embedding_model is None:
        return ReclusterResult(
            words=tuple(dict(w) for w in words),
            segments=tuple(dict(s) for s in segments),
            confidence=0.0,
            window_count=0,
            word_indices_per_window=(),
            changed=False,
        )

    # Phase 7b: Extract audio duration
    audio_duration_sec = len(audio) / float(sample_rate)

    # Phase 7c: Build embedding windows
    # Use Phase 5 defaults; caller may override via ReclusterConfig
    windows = build_embedding_windows(
        words=words,
        segments=segments,
        min_window_seconds=_DEFAULT_MIN_WINDOW_SEC,
        max_window_seconds=_DEFAULT_MAX_WINDOW_SEC,
        audio_duration_sec=audio_duration_sec,
        hop_on_speaker_boundaries=True,
    )

    if not windows:
        # No windows built → return unchanged
        return ReclusterResult(
            words=tuple(dict(w) for w in words),
            segments=tuple(dict(s) for s in segments),
            confidence=0.0,
            window_count=0,
            word_indices_per_window=(),
            changed=False,
        )

    # Phase 7d: Extract embeddings per window
    embeddings_list = []
    window_indices_per_word = [None] * len(words)

    for window_idx, window in enumerate(windows):
        # Extract audio for this window
        start_sample = int(window.start * sample_rate)
        end_sample = int(window.end * sample_rate)
        window_audio = audio[start_sample:end_sample].astype(np.float32)

        if len(window_audio) == 0:
            continue

        # Extract embedding
        embedding = embedding_model.extract_embedding(window_audio, sample_rate)

        if isinstance(embedding, EmbeddingUnavailable):
            # Embedding unavailable for this window → skip
            continue

        embeddings_list.append(embedding)

        # Map word indices to this window
        for word_idx in window.word_indices:
            window_indices_per_word[word_idx] = window_idx

    if not embeddings_list:
        # No embeddings extracted → return unchanged
        return ReclusterResult(
            words=tuple(dict(w) for w in words),
            segments=tuple(dict(s) for s in segments),
            confidence=0.0,
            window_count=len(windows),
            word_indices_per_window=tuple(w.word_indices for w in windows),
            changed=False,
        )

    # Phase 7e: Perform reclustering
    embeddings_array = np.vstack(embeddings_list)
    # confidence_threshold passed by caller via stt_processor config
    # Use Phase 6 default for now; will be overridden in integration
    updated_words, confidence, changed = recluster_speakers(
        words=words,
        embeddings=embeddings_array,
        window_indices_per_word=window_indices_per_word,
        confidence_threshold=_DEFAULT_CONFIDENCE_THRESHOLD,
    )

    # Phase 7f: Return immutable result
    return ReclusterResult(
        words=updated_words,
        segments=tuple(dict(s) for s in segments),  # segments typically unchanged
        confidence=confidence,
        window_count=len(windows),
        word_indices_per_window=tuple(w.word_indices for w in windows),
        changed=changed,
    )
