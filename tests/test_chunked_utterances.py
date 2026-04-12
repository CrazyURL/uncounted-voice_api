"""Integration tests for chunk-mode utterance WAV generation.

These tests cover the chunked pipeline's utterance splitting logic without
requiring GPU/WhisperX. They build chunk-local word inputs, call the shared
`utterance_segmenter.segment()` and `extract_utterance_audio_local()` paths,
and verify the per-chunk WAV emission + global timestamp composition that
`_transcribe_chunked` performs.

The production integration (T3/T4) is covered end-to-end on the GPU server
in Wave 4. Here we lock the pure-Python correctness of the chunk-local
timestamp convention and the global utterance index monotonicity.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.services.chunk_utterance_emitter import emit_chunk_utterances

SR = 16000


def _word(text: str, start: float, end: float, speaker: str = "SPEAKER_0") -> dict:
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def _make_chunk(duration_sec: float, fill: float = 1.0) -> np.ndarray:
    return np.full(int(duration_sec * SR), fill, dtype=np.float32)


def _as_segments(words: list[dict]) -> list[dict]:
    """Wrap chunk-local words as a single cleaned segment dict (matches
    the post-`_clean_segments` shape that `_transcribe_chunked` feeds in).
    """
    if not words:
        return []
    return [{
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w.get("word", "") for w in words),
        "speaker": words[0].get("speaker"),
        "words": words,
    }]


def _run_chunked_utterance_loop(
    chunk_plans: list[tuple[np.ndarray, float, list[dict]]],
) -> tuple[list[dict], dict[str, bytes]]:
    """Drive the real `_emit_chunk_utterances` across multiple chunks exactly
    as `_transcribe_chunked` does. Test input is chunk-local words per chunk.
    """
    all_utterances: list[dict] = []
    audio_files: dict[str, bytes] = {}
    global_idx = 0
    cumulative_offset = 0.0

    for chunk_audio, chunk_duration, chunk_words in chunk_plans:
        chunk_utts, chunk_files, global_idx = emit_chunk_utterances(
            chunk_audio,
            _as_segments(chunk_words),
            chunk_duration,
            cumulative_offset,
            global_idx,
            SR,
        )
        all_utterances.extend(chunk_utts)
        audio_files.update(chunk_files)
        cumulative_offset += chunk_duration

    return all_utterances, audio_files


class TestChunkedUtteranceLoop:
    def test_emits_utterances_across_two_chunks(self):
        chunk_a = _make_chunk(30.0, fill=0.1)
        chunk_b = _make_chunk(30.0, fill=0.2)
        # Chunk A: two utterances at 2-5s and 10-14s (chunk-local)
        words_a = [
            _word("hello", 2.0, 2.5),
            _word("world", 2.5, 5.0),
            _word("second", 10.0, 12.0),
            _word("utterance", 12.0, 14.0),
        ]
        # Chunk B: one utterance at 3-7s (chunk-local)
        words_b = [
            _word("third", 3.0, 5.0, "SPEAKER_1"),
            _word("segment", 5.0, 7.0, "SPEAKER_1"),
        ]

        utts, files = _run_chunked_utterance_loop([
            (chunk_a, 30.0, words_a),
            (chunk_b, 30.0, words_b),
        ])

        assert len(utts) >= 2
        assert len(files) == len(utts)
        # Global index is monotonic starting at 0
        assert [u["index"] for u in utts] == list(range(len(utts)))

    def test_global_timestamps_include_cumulative_offset(self):
        chunk_a = _make_chunk(30.0)
        chunk_b = _make_chunk(30.0)
        # Chunk A has one utterance, chunk B has one utterance at local 5-9s
        words_a = [_word("first", 1.0, 3.0)]
        words_b = [_word("later", 5.0, 9.0, "SPEAKER_1")]

        utts, _ = _run_chunked_utterance_loop([
            (chunk_a, 30.0, words_a),
            (chunk_b, 30.0, words_b),
        ])

        # Second chunk's utterance should be globalized by +30.0
        assert len(utts) == 2
        assert utts[1]["start_sec"] >= 30.0 + 5.0 - 0.5  # allow padding
        assert utts[1]["start_sec"] < 30.0 + 5.0 + 0.5
        assert utts[1]["end_sec"] >= 30.0 + 9.0 - 0.5
        assert utts[1]["end_sec"] <= 30.0 + 9.0 + 0.5

    def test_wav_bytes_have_valid_riff_header(self):
        chunk = _make_chunk(20.0)
        words = [_word("valid", 2.0, 4.0), _word("wav", 4.0, 6.0)]

        utts, files = _run_chunked_utterance_loop([(chunk, 20.0, words)])

        assert len(files) == len(utts)
        for filename, data in files.items():
            assert data[:4] == b"RIFF"
            assert data[8:12] == b"WAVE"
            # Soundfile should be able to decode the bytes back
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            assert sr == SR
            assert len(audio) > 0

    def test_filenames_and_indices_monotonic_across_many_chunks(self):
        chunks = []
        for i in range(4):
            chunks.append((
                _make_chunk(20.0),
                20.0,
                [_word(f"chunk{i}", 2.0, 4.0)],
            ))

        utts, files = _run_chunked_utterance_loop(chunks)

        assert len(utts) == 4
        assert [u["audio_filename"] for u in utts] == [
            "utterance_000.wav",
            "utterance_001.wav",
            "utterance_002.wav",
            "utterance_003.wav",
        ]
        assert set(files.keys()) == {u["audio_filename"] for u in utts}

    def test_no_utterances_when_no_words(self):
        utts, files = _run_chunked_utterance_loop([
            (_make_chunk(10.0), 10.0, []),
            (_make_chunk(10.0), 10.0, []),
        ])
        assert utts == []
        assert files == {}

    def test_boundary_word_at_chunk_end_clamped_by_total_duration(self):
        # Word extends right up to the chunk duration — padding should be
        # clamped to chunk_duration and WAV should still be produced
        chunk = _make_chunk(15.0)
        words = [_word("edge", 13.5, 14.9)]

        utts, files = _run_chunked_utterance_loop([(chunk, 15.0, words)])

        assert len(utts) == 1
        # Global end should not exceed chunk_duration + 0 (cumulative_offset=0)
        assert utts[0]["end_sec"] <= 15.0 + 0.01

    def test_does_not_mutate_input_chunk(self):
        chunk = _make_chunk(20.0, fill=0.5)
        original = chunk.copy()
        words = [_word("hi", 2.0, 4.0)]

        _run_chunked_utterance_loop([(chunk, 20.0, words)])

        assert np.array_equal(chunk, original)
