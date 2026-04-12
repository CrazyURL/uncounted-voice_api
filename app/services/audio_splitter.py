"""Audio splitting utilities — extract speaker/utterance audio from PCM."""

import io

import numpy as np
import soundfile as sf

from app.services.utterance_segmenter import UtteranceBoundary


def extract_speaker_audio(
    audio: np.ndarray,
    segments: list[dict],
    speaker_id: str,
    sr: int,
) -> np.ndarray | None:
    """Concatenate audio regions belonging to a specific speaker.

    Uses word-level boundaries when available for precise extraction.
    Falls back to segment-level if words are not present.
    Returns None if total duration < 0.5s.
    """
    ranges: list[tuple[int, int]] = []

    for seg in segments:
        words = seg.get("words")
        if words:
            # Word-level: extract only words matching this speaker
            for w in words:
                w_speaker = w.get("speaker")
                if w_speaker != speaker_id:
                    continue
                start_sample = max(0, min(int(w.get("start", 0) * sr), len(audio)))
                end_sample = max(0, min(int(w.get("end", 0) * sr), len(audio)))
                if end_sample > start_sample:
                    ranges.append((start_sample, end_sample))
        else:
            # Segment-level fallback
            seg_speaker = seg.get("speaker", seg.get("speakerId"))
            if seg_speaker != speaker_id:
                continue
            start_sample = max(0, min(int(seg.get("start", 0) * sr), len(audio)))
            end_sample = max(0, min(int(seg.get("end", 0) * sr), len(audio)))
            if end_sample > start_sample:
                ranges.append((start_sample, end_sample))

    if not ranges:
        return None

    ranges.sort(key=lambda r: r[0])
    merged = _merge_ranges(ranges)

    total_samples = sum(e - s for s, e in merged)
    if total_samples < sr // 2:
        return None

    parts = [audio[s:e] for s, e in merged]
    return np.concatenate(parts)


def extract_utterance_audio(
    audio: np.ndarray,
    utterance: UtteranceBoundary,
    sr: int,
) -> np.ndarray:
    """Slice audio for an utterance boundary (using padded times)."""
    start = max(0, int(utterance.padded_start_sec * sr))
    end = min(len(audio), int(utterance.padded_end_sec * sr))
    return audio[start:end]


def extract_utterance_audio_local(
    chunk_audio: np.ndarray,
    padded_start_local_sec: float,
    padded_end_local_sec: float,
    sr: int,
) -> np.ndarray:
    """Slice a chunk-local region of `chunk_audio` by second offsets.

    Used in chunked mode, where `chunk_audio` only holds the current chunk
    and timestamps from utterance_segmenter are relative to the chunk start.
    Boundaries are clipped to `[0, len(chunk_audio)]`. Returns an empty
    array (with matching dtype) when the clipped range is empty.
    """
    start = max(0, int(padded_start_local_sec * sr))
    end = min(len(chunk_audio), int(padded_end_local_sec * sr))
    if end <= start:
        return chunk_audio[0:0]
    return chunk_audio[start:end]


def mute_non_speaker(
    audio: np.ndarray,
    segments: list[dict],
    keep_speaker_id: str,
    sr: int,
) -> np.ndarray:
    """Zero out audio regions not belonging to the specified speaker.

    Uses word-level boundaries when available for precise muting.
    Falls back to segment-level if words are not present.
    """
    muted = audio.copy()
    for seg in segments:
        words = seg.get("words")
        if words:
            for w in words:
                w_speaker = w.get("speaker")
                if w_speaker == keep_speaker_id or w_speaker is None:
                    continue
                start = max(0, min(int(w.get("start", 0) * sr), len(muted)))
                end = max(0, min(int(w.get("end", 0) * sr), len(muted)))
                muted[start:end] = 0.0
        else:
            seg_speaker = seg.get("speaker", seg.get("speakerId"))
            if seg_speaker == keep_speaker_id:
                continue
            start = max(0, min(int(seg.get("start", 0) * sr), len(muted)))
            end = max(0, min(int(seg.get("end", 0) * sr), len(muted)))
            muted[start:end] = 0.0
    return muted


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert numpy audio array to WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping sample ranges."""
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
