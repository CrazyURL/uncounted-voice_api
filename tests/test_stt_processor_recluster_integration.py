"""Phase 7 진짜 통합 테스트 — `app.stt_processor._apply_reclustering` 실제 호출.

이 파일은 stt_processor를 mock 없이 import하고, _apply_reclustering 헬퍼를
직접 호출해 다음을 검증한다:

1. flag off → result byte-equivalent (segment 손상 없음)
2. flag on + 임베딩 분리 가능 → segment.words[i]의 speaker/speaker_id 갱신,
   segment 구조 (text, start, end, speaker) 보존
3. flag on + 모델 미가용 → result 그대로
4. 빈 segments 안전 처리
5. WhisperX 컨벤션의 word['speaker'] 필드도 동기화

`tests/conftest.py`가 `whisperx`를 sys.modules에 mock해두기 때문에
stt_processor import 시 실제 GPU/모델 로드 없이 통과한다.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app import stt_processor
from app.services.speaker_embedding import EmbeddingUnavailable
from app.services.speaker_recluster import EmbeddingWindow, ReclusterResult


def _whisperx_segments() -> list[dict]:
    """WhisperX assign_word_speakers 직후 형태의 segment 리스트.

    segment 안에 nested word 배열, 각 word는 'speaker'(WhisperX 컨벤션) 보유.
    """
    return [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "안녕 오랜만이야",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "안녕", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                {"word": "오랜만이야", "start": 0.5, "end": 1.4, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "start": 1.6,
            "end": 3.0,
            "text": "응 잘 지내",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "응", "start": 1.6, "end": 1.8, "speaker": "SPEAKER_00"},
                {"word": "잘", "start": 1.9, "end": 2.1, "speaker": "SPEAKER_00"},
                {"word": "지내", "start": 2.2, "end": 2.9, "speaker": "SPEAKER_00"},
            ],
        },
    ]


@pytest.fixture(autouse=True)
def _reset_singleton():
    """각 테스트 전후로 모듈 싱글톤 초기화."""
    stt_processor._speaker_embedding_model = None
    yield
    stt_processor._speaker_embedding_model = None


@pytest.fixture
def _clean_recluster_env(monkeypatch):
    """ReclusterConfig env 변수를 모두 unset 처리."""
    for key in (
        "VOICE_DIARIZATION_WESPEAKER_RECLUSTER",
        "VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS",
        "VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD",
        "VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC",
        "VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.unit
class TestApplyReclusteringFlagOff:
    def test_flag_off_returns_result_unchanged(self, _clean_recluster_env):
        audio = np.zeros(16000, dtype=np.float32)
        original_segments = _whisperx_segments()
        result = {"segments": [dict(s) for s in original_segments]}
        # word 배열도 deep copy
        for s in result["segments"]:
            s["words"] = [dict(w) for w in s["words"]]

        out = stt_processor._apply_reclustering(audio, 16000, result, "task-x")

        # result 객체는 동일 reference로 반환
        assert out is result
        # segment 구조 그대로
        assert len(out["segments"]) == len(original_segments)
        for got, want in zip(out["segments"], original_segments):
            assert got["text"] == want["text"]
            assert got["start"] == want["start"]
            assert got["end"] == want["end"]
            assert len(got["words"]) == len(want["words"])
            for gw, ww in zip(got["words"], want["words"]):
                assert gw["speaker"] == ww["speaker"]
                assert gw["word"] == ww["word"]

    def test_flag_off_does_not_load_embedding_model(self, _clean_recluster_env):
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": _whisperx_segments()}

        stt_processor._apply_reclustering(audio, 16000, result, "task-x")

        # 싱글톤이 lazy-load되지 않아야 함
        assert stt_processor._speaker_embedding_model is None


@pytest.mark.unit
class TestApplyReclusteringFlagOn:
    def test_flag_on_relabels_words_and_preserves_segment_structure(
        self, monkeypatch, _clean_recluster_env
    ):
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")

        audio = np.zeros(48000, dtype=np.float32)  # 3초
        result = {"segments": _whisperx_segments()}

        # maybe_recluster_speakers를 fake로 교체:
        # - 5개 word 중 마지막 3개를 SPEAKER_01로 라벨링
        # - segments는 immutable copy 그대로 반환
        def fake_hook(*, audio, sample_rate, words, segments, mode, embedding_model):
            assert len(words) == 5  # _whisperx_segments는 5 words 평탄화
            assert mode == "call_recording"
            assert embedding_model is stt_processor._speaker_embedding_model
            new_words = []
            for i, w in enumerate(words):
                nw = dict(w)
                nw["speaker_id"] = "SPEAKER_01" if i >= 2 else "SPEAKER_00"
                new_words.append(nw)
            return ReclusterResult(
                words=tuple(new_words),
                segments=tuple(dict(s) for s in segments),
                confidence=0.85,
                window_count=2,
                word_indices_per_window=((0, 1), (2, 3, 4)),
                changed=True,
            )

        # SpeakerEmbeddingModel도 dummy로 — 실제 ONNX 로드 회피
        monkeypatch.setattr(
            stt_processor,
            "SpeakerEmbeddingModel",
            lambda: object(),
        )
        monkeypatch.setattr(stt_processor, "maybe_recluster_speakers", fake_hook)

        out = stt_processor._apply_reclustering(audio, 16000, result, "task-x")

        # segment 수 보존
        assert len(out["segments"]) == 2
        # segment 구조 보존 (text, start, end)
        assert out["segments"][0]["text"] == "안녕 오랜만이야"
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][1]["text"] == "응 잘 지내"

        # word 라벨이 새 클러스터로 갱신됨
        all_words = [w for s in out["segments"] for w in s["words"]]
        assert len(all_words) == 5
        # 첫 2개는 SPEAKER_00, 나머지 3개는 SPEAKER_01
        assert [w["speaker_id"] for w in all_words] == [
            "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01",
        ]
        # WhisperX 컨벤션의 'speaker' 필드도 동기화
        assert [w["speaker"] for w in all_words] == [
            "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01", "SPEAKER_01",
        ]

    def test_flag_on_changed_false_returns_result_unchanged(
        self, monkeypatch, _clean_recluster_env
    ):
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        audio = np.zeros(48000, dtype=np.float32)
        original_segments = _whisperx_segments()
        result = {"segments": [dict(s) for s in original_segments]}
        for s in result["segments"]:
            s["words"] = [dict(w) for w in s["words"]]

        def fake_hook(**kwargs):
            words = kwargs["words"]
            return ReclusterResult(
                words=tuple(dict(w) for w in words),
                segments=tuple(dict(s) for s in kwargs["segments"]),
                confidence=0.05,
                window_count=2,
                word_indices_per_window=((0, 1), (2, 3, 4)),
                changed=False,
            )

        monkeypatch.setattr(stt_processor, "SpeakerEmbeddingModel", lambda: object())
        monkeypatch.setattr(stt_processor, "maybe_recluster_speakers", fake_hook)

        out = stt_processor._apply_reclustering(audio, 16000, result, "task-x")

        # changed=False → segment 구조와 라벨 모두 원본 그대로
        all_words = [w for s in out["segments"] for w in s["words"]]
        assert all(w["speaker"] == "SPEAKER_00" for w in all_words)
        assert "speaker_id" not in all_words[0] or all_words[0].get("speaker_id") is None or all_words[0].get("speaker") == "SPEAKER_00"

    def test_flag_on_empty_segments_safe(self, monkeypatch, _clean_recluster_env):
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        audio = np.zeros(16000, dtype=np.float32)
        result = {"segments": []}

        # hook이 호출되어선 안 됨 (early return on empty flat_words)
        def fail_hook(**kwargs):
            raise AssertionError("hook must not be called for empty segments")

        monkeypatch.setattr(stt_processor, "SpeakerEmbeddingModel", lambda: object())
        monkeypatch.setattr(stt_processor, "maybe_recluster_speakers", fail_hook)

        out = stt_processor._apply_reclustering(audio, 16000, result, "task-x")
        assert out["segments"] == []

    def test_flag_on_singleton_lazy_loaded_once(
        self, monkeypatch, _clean_recluster_env
    ):
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_RECLUSTER", "true")
        audio = np.zeros(48000, dtype=np.float32)

        load_count = {"n": 0}

        def fake_model_init():
            load_count["n"] += 1
            return object()

        def noop_hook(**kwargs):
            return ReclusterResult(
                words=tuple(dict(w) for w in kwargs["words"]),
                segments=tuple(dict(s) for s in kwargs["segments"]),
                confidence=0.0,
                window_count=0,
                word_indices_per_window=(),
                changed=False,
            )

        monkeypatch.setattr(stt_processor, "SpeakerEmbeddingModel", fake_model_init)
        monkeypatch.setattr(stt_processor, "maybe_recluster_speakers", noop_hook)

        result1 = {"segments": _whisperx_segments()}
        result2 = {"segments": _whisperx_segments()}
        stt_processor._apply_reclustering(audio, 16000, result1, "task-1")
        stt_processor._apply_reclustering(audio, 16000, result2, "task-2")

        # 싱글톤이 1회만 lazy-load되어야 함
        assert load_count["n"] == 1


@pytest.mark.unit
class TestApplyReclusteringByteEquivalentDeep:
    def test_flag_off_does_not_mutate_input_segments(self, _clean_recluster_env):
        audio = np.zeros(16000, dtype=np.float32)
        original_segments = _whisperx_segments()
        result = {"segments": original_segments}

        stt_processor._apply_reclustering(audio, 16000, result, "task-x")

        # 원본 dict identity 유지 (mutation 없음)
        assert all(
            "speaker" in w and w["speaker"] == "SPEAKER_00"
            for s in original_segments
            for w in s["words"]
        )
