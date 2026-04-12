"""Pure helpers for emitting utterance WAVs from a single chunk.

Extracted from `stt_processor` so unit tests can import them without pulling
in the GPU / whisperx module graph. No GPU, torch, or I/O dependencies here.
"""
from __future__ import annotations

import numpy as np

from app.services.audio_splitter import (
    extract_utterance_audio_local,
    to_wav_bytes,
)
from app.services.utterance_segmenter import segment as segment_utterances


def collect_words_with_speaker_fallback(segments: list[dict]) -> list[dict]:
    """청크 단위 segments → (word, start, end, speaker) 플랫 리스트.

    `speaker`가 비어 있으면 인접 단어에서 전파해 `SPEAKER_0`으로 폴백한다.
    """
    words: list[dict] = []
    for s in segments:
        if s.get("words"):
            for w in s["words"]:
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", s["start"]),
                    "end": w.get("end", s["end"]),
                    "speaker": w.get("speaker", s.get("speaker")),
                })
        else:
            words.append({
                "word": s.get("text", ""),
                "start": s["start"],
                "end": s["end"],
                "speaker": s.get("speaker"),
            })

    for i, w in enumerate(words):
        if w["speaker"] is None:
            if i > 0 and words[i - 1]["speaker"] is not None:
                w["speaker"] = words[i - 1]["speaker"]
            elif i + 1 < len(words) and words[i + 1]["speaker"] is not None:
                w["speaker"] = words[i + 1]["speaker"]
            else:
                w["speaker"] = "SPEAKER_0"
    return words


def emit_chunk_utterances(
    chunk_audio: np.ndarray,
    chunk_segments: list[dict],
    preprocessed_chunk_duration: float,
    cumulative_offset: float,
    start_global_idx: int,
    sr: int,
) -> tuple[list[dict], dict[str, bytes], int]:
    """청크 내 발화를 분리해 WAV를 생성하고 글로벌 타임스탬프로 반환한다.

    순수 함수. `chunk_audio`가 메모리에 상주한 상태에서만 호출한다. 호출자는
    반환된 `next_global_idx`로 카운터를 유지해 청크 간 단조 증가를 보장한다.
    """
    chunk_local_words = collect_words_with_speaker_fallback(chunk_segments)
    if not chunk_local_words:
        return [], {}, start_global_idx

    chunk_local_utts = segment_utterances(chunk_local_words, preprocessed_chunk_duration)
    utterances: list[dict] = []
    audio_files: dict[str, bytes] = {}
    idx = start_global_idx

    for utt in chunk_local_utts:
        utt_audio = extract_utterance_audio_local(
            chunk_audio, utt.padded_start_sec, utt.padded_end_sec, sr,
        )
        if len(utt_audio) == 0:
            continue
        filename = f"utterance_{idx:03d}.wav"
        audio_files[filename] = to_wav_bytes(utt_audio, sr)
        utterances.append({
            "index": idx,
            "start_sec": round(utt.start_sec + cumulative_offset, 2),
            "end_sec": round(utt.end_sec + cumulative_offset, 2),
            "duration_sec": utt.duration_sec,
            "speaker_id": utt.speaker_id,
            "transcript_text": utt.transcript_text,
            "audio_filename": filename,
            "words": [
                {
                    **w,
                    "start": round(w.get("start", 0) + cumulative_offset, 2),
                    "end": round(w.get("end", 0) + cumulative_offset, 2),
                }
                for w in utt.words
            ],
        })
        idx += 1

    return utterances, audio_files, idx
