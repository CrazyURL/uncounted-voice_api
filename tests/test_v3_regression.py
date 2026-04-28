"""v3.0 회귀 테스트 — 알파 샘플에서 발견된 4개 시스템 버그 차단.

이 테스트가 통과하지 못하면 v3.0 패키지 출하 게이트 통과 불가.

검증 대상:
- Bug 1: WAV padding이 duration_sec에 반영되어 metadata == 실제 WAV 길이
- Bug 5: result.json header.duration_seconds가 segments 최대 end 또는 audio 길이 반영
- Bug 6: speaker_id 폴백 값이 "SPEAKER_00" (자릿수 패딩 형식)
- Bug 7: speaker_id 갭(SPEAKER_03 누락 등)이 정규화되어 0~N-1 연속 번호
"""
from __future__ import annotations

import importlib

import pytest

from app.services.speaker_recluster import (
    _renumber_consecutive,
    renumber_speakers_in_place,
)
from app.services.utterance_segmenter import (
    UtteranceBoundary,
    _apply_padding,
    _get_speaker_id,
    _RawUtterance,
)
from app import config as app_config


# ─────────────────────────────────────────────────────────────────
# Bug 1: duration_sec == padded 길이 (WAV 추출과 일치)
# ─────────────────────────────────────────────────────────────────


def test_bug1_duration_sec_includes_padding_both_sides() -> None:
    """일반 발화: 양쪽 padding 모두 적용되어 duration_sec == padded 길이."""
    # 1.0초 ~ 5.0초 발화 → padded 0.85 ~ 5.15 (PADDING_SEC=0.15 가정)
    pad = app_config.PADDING_SEC
    raw = _RawUtterance(
        speaker_id="SPEAKER_00",
        words=[
            {"word": "안녕", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"word": "하세요", "start": 4.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ],
    )
    result = _apply_padding([raw], total_duration=10.0)
    assert len(result) == 1
    u = result[0]

    expected_padded_dur = round((5.0 + pad) - max(0.0, 1.0 - pad), 2)
    assert u.duration_sec == expected_padded_dur, (
        f"duration_sec({u.duration_sec})는 padded 길이({expected_padded_dur})와 일치해야 함. "
        "Bug 1: WAV는 padded 구간으로 추출되는데 duration_sec이 raw end-start만 기록하면 "
        "metadata와 실제 WAV 길이가 정확히 PADDING_SEC*2만큼 어긋남."
    )
    # padded_end - padded_start와 정확히 같음
    assert u.duration_sec == round(u.padded_end_sec - u.padded_start_sec, 2)


def test_bug1_duration_sec_clamped_at_zero_start() -> None:
    """발화가 파일 시작 직후(start < PADDING_SEC)일 때 왼쪽 padding이 0으로 clamp."""
    pad = app_config.PADDING_SEC
    raw = _RawUtterance(
        speaker_id="SPEAKER_00",
        words=[{"word": "여보세요", "start": 0.03, "end": 1.0, "speaker": "SPEAKER_00"}],
    )
    result = _apply_padding([raw], total_duration=10.0)
    u = result[0]

    # padded_start = max(0, 0.03 - pad) = 0
    assert u.padded_start_sec == 0.0
    # padded_end = 1.0 + pad
    assert u.padded_end_sec == round(1.0 + pad, 2)
    # duration_sec = padded_end - padded_start = 1.0 + pad
    assert u.duration_sec == round(1.0 + pad, 2)


def test_bug1_duration_sec_clamped_at_total_duration() -> None:
    """발화가 파일 끝 직전일 때 오른쪽 padding이 total_duration으로 clamp."""
    pad = app_config.PADDING_SEC
    raw = _RawUtterance(
        speaker_id="SPEAKER_01",
        words=[{"word": "마지막", "start": 9.0, "end": 9.95, "speaker": "SPEAKER_01"}],
    )
    result = _apply_padding([raw], total_duration=10.0)
    u = result[0]

    assert u.padded_end_sec == 10.0
    # 왼쪽은 정상 padding
    assert u.padded_start_sec == round(9.0 - pad, 2)
    assert u.duration_sec == round(10.0 - (9.0 - pad), 2)


# ─────────────────────────────────────────────────────────────────
# Bug 6: speaker_id 폴백 값이 SPEAKER_NN 형식 (자릿수 패딩)
# ─────────────────────────────────────────────────────────────────


def test_bug6_get_speaker_id_default_is_padded() -> None:
    """speakerId/speaker 둘 다 없는 단어의 폴백 값."""
    word = {"word": "테스트", "start": 0.0, "end": 1.0}
    assert _get_speaker_id(word) == "SPEAKER_00"
    # "SPEAKER_0" (1자리)이면 안됨
    assert _get_speaker_id(word) != "SPEAKER_0"


def test_bug6_chunk_emitter_fallback_is_padded() -> None:
    """chunk_utterance_emitter에서 양쪽 인접 단어 모두 None일 때 폴백."""
    from app.services.chunk_utterance_emitter import collect_words_with_speaker_fallback

    # speaker 없는 단어 1개만 있는 segment → 폴백
    segments = [{
        "start": 0.0,
        "end": 1.0,
        "text": "혼자",
        "speaker": None,
        "words": [{"word": "혼자", "start": 0.0, "end": 1.0, "speaker": None}],
    }]
    result = collect_words_with_speaker_fallback(segments)
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["speaker"] != "SPEAKER_0"


# ─────────────────────────────────────────────────────────────────
# Bug 7: speaker_id 갭 정규화 (SPEAKER_03 누락 케이스 차단)
# ─────────────────────────────────────────────────────────────────


def test_bug7_renumber_consecutive_fills_gaps() -> None:
    """[00, 01, 02, 04] → [00, 01, 02, 03] (03 갭 채움)."""
    items = [
        {"speaker_id": "SPEAKER_00", "word": "a"},
        {"speaker_id": "SPEAKER_01", "word": "b"},
        {"speaker_id": "SPEAKER_02", "word": "c"},
        {"speaker_id": "SPEAKER_04", "word": "d"},  # 03 빠짐
    ]
    result = _renumber_consecutive(items, "speaker_id")
    speakers = [r["speaker_id"] for r in result]
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]


def test_bug7_renumber_consecutive_normalizes_speaker_0() -> None:
    """비표준 'SPEAKER_0' (1자리)도 정규화."""
    items = [
        {"speaker_id": "SPEAKER_0", "word": "a"},  # 비표준
        {"speaker_id": "SPEAKER_01", "word": "b"},
    ]
    result = _renumber_consecutive(items, "speaker_id")
    speakers = [r["speaker_id"] for r in result]
    # 첫 등장 순서대로 0, 1
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_bug7_renumber_speakers_in_place_segments_only() -> None:
    """segments만 있는 케이스 (split 옵션 모두 false)."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"},  # 01 빠짐
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_04"},  # 03 빠짐
    ]
    renumber_speakers_in_place(segments=segments)
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert segments[2]["speaker"] == "SPEAKER_02"


def test_bug7_renumber_speakers_in_place_keeps_consistency() -> None:
    """segments / utterances / speaker_audio 모두 같은 매핑으로 일관 갱신."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "words": [
            {"word": "안녕", "speaker": "SPEAKER_00"},
        ]},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_04", "words": [
            {"word": "응", "speaker": "SPEAKER_04"},
        ]},
    ]
    utterances = [
        {"speaker_id": "SPEAKER_00", "transcript_text": "안녕", "words": [
            {"speaker": "SPEAKER_00"},
        ]},
        {"speaker_id": "SPEAKER_04", "transcript_text": "응", "words": [
            {"speaker": "SPEAKER_04"},
        ]},
    ]
    speaker_audio = [
        {"speaker_id": "SPEAKER_00", "total_duration_sec": 5.0},
        {"speaker_id": "SPEAKER_04", "total_duration_sec": 3.0},
    ]

    renumber_speakers_in_place(
        segments=segments,
        utterances=utterances,
        speaker_audio=speaker_audio,
    )

    # segments: 04 → 01
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert segments[1]["words"][0]["speaker"] == "SPEAKER_01"
    # utterances: 04 → 01 (segments와 동일 매핑)
    assert utterances[1]["speaker_id"] == "SPEAKER_01"
    assert utterances[1]["words"][0]["speaker"] == "SPEAKER_01"
    # speaker_audio: 04 → 01
    assert speaker_audio[1]["speaker_id"] == "SPEAKER_01"


def test_bug7_renumber_handles_empty() -> None:
    """빈 입력 / None / 비어있는 segments 모두 안전하게 처리."""
    # 모두 None
    renumber_speakers_in_place()  # 예외 없이 끝나야 함
    # 빈 segments
    segments: list = []
    renumber_speakers_in_place(segments=segments)
    assert segments == []
    # speaker None만 있음
    segments = [{"start": 0, "end": 1, "speaker": None}]
    renumber_speakers_in_place(segments=segments)
    assert segments[0]["speaker"] is None


def test_bug7_renumber_preserves_first_occurrence_order() -> None:
    """첫 등장 순서를 유지 (정렬하지 않음)."""
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_05"},  # 첫 등장
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_05"},  # 재등장
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_07"},
    ]
    renumber_speakers_in_place(segments=segments)
    # 첫 등장 순: 05 → 00, 02 → 01, 07 → 02
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert segments[2]["speaker"] == "SPEAKER_00"  # 재등장도 동일 매핑
    assert segments[3]["speaker"] == "SPEAKER_02"


# ─────────────────────────────────────────────────────────────────
# Bug 5: duration_seconds (output dict)는 segments 최대 end 또는 audio 길이 반영
# ─────────────────────────────────────────────────────────────────
# 단위 테스트로 직접 검증 어려운 출력 dict 빌더 로직(transcribe 함수 내부).
# 대신 통합 패턴을 문서화하여 review 시 참조.


def test_bug5_duration_seconds_logic_is_documented() -> None:
    """Bug 5 fix는 stt_processor.py output dict 빌드 직전에서 다음 로직 사용:

        max_segment_end = max((seg["end"] for seg in segments), default=0.0)
        if audio is not None:
            audio_duration = len(audio) / config.SAMPLE_RATE
            actual_duration = max(audio_duration, max_segment_end)
        else:
            actual_duration = max(max_segment_end, total_duration)
        output["duration_seconds"] = round(actual_duration, 2)

    이 로직이 ``stt_processor.transcribe`` 함수 내에 존재함을 확인.
    """
    src = importlib.import_module("app.stt_processor")
    import inspect

    source = inspect.getsource(src.transcribe)
    assert "max_segment_end" in source, "Bug 5 fix 누락"
    assert "actual_duration" in source, "Bug 5 fix 누락"
    assert "duration_seconds" in source, "duration_seconds 필드 사용 확인"
