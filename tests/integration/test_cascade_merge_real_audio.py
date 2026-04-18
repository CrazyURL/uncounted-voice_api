"""Integration test — cascade merge 버그 회귀 검증 (실제 WAV 파일 + 직접 파이프라인 호출).

utt_34c810c30650d12e_008.wav (23.81s) 를 stt_processor.transcribe() 로 직접 처리하여
발화 분리가 제대로 이루어지는지 확인한다.

실행 방법:
    RUN_GPU_INTEGRATION=1 \
    HF_TOKEN=<token> \
    REAL_WAV_PATH=<path>/utt_34c810c30650d12e_008.wav \
    pytest tests/integration/test_cascade_merge_real_audio.py -v

환경변수:
    RUN_GPU_INTEGRATION  — "1" 이어야 실행 (기본: skip)
    HF_TOKEN             — pyannote 화자분리 필수
    REAL_WAV_PATH        — 테스트할 WAV 파일 경로 (기본: downloads/utterances/.../008.wav)
"""

import os
from pathlib import Path

import pytest

_DEFAULT_WAV = str(
    Path(__file__).parents[3]
    / "uncounted-api/downloads/utterances/34c810c30650d12e/utt_34c810c30650d12e_008.wav"
)
REAL_WAV = os.environ.get("REAL_WAV_PATH", _DEFAULT_WAV)


def _skip_reason() -> str | None:
    if os.environ.get("RUN_GPU_INTEGRATION") != "1":
        return "RUN_GPU_INTEGRATION != '1'"
    if not os.environ.get("HF_TOKEN"):
        return "HF_TOKEN not set (required for pyannote)"
    if not Path(REAL_WAV).exists():
        return f"WAV 파일 없음: {REAL_WAV}"
    return None


_reason = _skip_reason()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.real_audio,
    pytest.mark.skipif(_reason is not None, reason=_reason or ""),
]


@pytest.fixture(scope="module")
def stt_result(tmp_path_factory):
    """008.wav 를 직접 transcribe() 로 처리한 결과 (모듈 단위 캐시)."""
    from app.stt_processor import load_models, transcribe

    load_models()

    return transcribe(
        file_path=REAL_WAV,
        task_id="test_cascade_merge_008",
        enable_diarize=True,
        enable_name_masking=False,
        mask_pii=False,
        split_by_speaker=False,
        split_by_utterance=True,
        denoise_enabled=False,
    )


class TestCascadeMergeRealAudio:
    """utt_34c810c30650d12e_008.wav — 23.81s 과병합 버그 회귀 검증."""

    def test_utterances_present(self, stt_result):
        """split_by_utterance=True 시 utterances 필드가 있어야 한다."""
        assert "utterances" in stt_result
        assert stt_result["utterances"]

    def test_splits_into_multiple_utterances(self, stt_result):
        """23.81s 파일이 2개 이상의 utterance 로 분할돼야 한다.

        버그 상태: _merge_short_utterances 무제한 누적 → 1개 (23.81s)
        픽스 상태: last.duration >= MIN_UTTERANCE_SEC 시 병합 중단 → 복수 utterances
        """
        utterances = stt_result["utterances"]
        durations = [(u["speaker_id"], u["duration_sec"]) for u in utterances]
        assert len(utterances) >= 2, (
            f"cascade merge 버그 재현: {len(utterances)}개 utterance\n"
            f"발화 목록: {durations}"
        )

    def test_no_utterance_exceeds_max(self, stt_result):
        """어떤 utterance 도 MAX_UTTERANCE_SEC 를 초과하면 안 된다."""
        from app import config
        for u in stt_result["utterances"]:
            assert u["duration_sec"] <= config.MAX_UTTERANCE_SEC, (
                f"MAX_UTTERANCE_SEC({config.MAX_UTTERANCE_SEC}s) 초과: "
                f"{u['duration_sec']:.2f}s [{u['speaker_id']}] '{u['transcript_text'][:40]}'"
            )

    def test_audio_duration_matches(self, stt_result):
        """인식된 총 오디오 길이가 원본(23.81s) 과 ±1s 오차 내여야 한다."""
        assert abs(stt_result["duration_seconds"] - 23.81) < 1.0, (
            f"길이 불일치: {stt_result['duration_seconds']:.2f}s (예상 ~23.81s)"
        )
