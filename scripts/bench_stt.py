"""WhisperX STT 벤치마크 — baseline vs turbo 비교용.

사용법 (WSL, venv 활성화 상태):
    # voice-api@dev 중단 후 실행 필수 (GPU 중복 로드 방지)
    sudo systemctl stop voice-api@dev

    cd ~/project/Uncounted-root/uncounted-voice-api
    source venv/bin/activate

    # baseline (large-v3)
    python scripts/bench_stt.py \
        --samples ~/voice-api-bench/samples.json \
        --audio-root ~/voice-api-bench \
        --out ~/voice-api-bench/results/baseline \
        --label baseline

    # turbo (MODEL_SIZE 환경변수 덮어쓰기 후 재실행)
    MODEL_SIZE=large-v3-turbo python scripts/bench_stt.py \
        --samples ~/voice-api-bench/samples.json \
        --audio-root ~/voice-api-bench \
        --out ~/voice-api-bench/results/turbo \
        --label turbo

산출물:
    results/<label>/<hash>.result.json   — stt_processor.transcribe() 반환값
    results/<label>/<hash>.meta.json     — 샘플별 측정 메타 (elapsed, RTF, VRAM)
    results/<label>/summary.json         — 전체 요약
    results/<label>/run.log              — 실행 로그
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


def _strip_bytes(obj: Any) -> Any:
    """dict/list 트리에서 bytes 값을 제거해 JSON 직렬화 가능하게 만든다."""
    if isinstance(obj, dict):
        return {k: _strip_bytes(v) for k, v in obj.items() if not isinstance(v, (bytes, bytearray))}
    if isinstance(obj, list):
        return [_strip_bytes(v) for v in obj if not isinstance(v, (bytes, bytearray))]
    return obj

# 경로 설정: app.* import 가능하도록 프로젝트 루트 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, type=Path, help="samples.json 경로")
    p.add_argument("--audio-root", required=True, type=Path, help="오디오 파일 루트 (samples.file의 기준)")
    p.add_argument("--out", required=True, type=Path, help="결과 출력 디렉토리")
    p.add_argument("--label", required=True, help="측정 라벨 (예: baseline, turbo)")
    p.add_argument("--skip-diarize", action="store_true", help="화자분리 비활성 (pyannote 미적용 측정)")
    return p.parse_args()


class VramSampler(threading.Thread):
    """nvidia-smi 로 GPU VRAM 사용량을 주기 샘플링."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval_sec
        self._stop_event = threading.Event()
        self.samples: list[int] = []  # MiB 단위

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=2,
                )
                mib = int(out.strip().splitlines()[0])
                self.samples.append(mib)
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
    )


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out / "run.log")
    logger = logging.getLogger("bench_stt")

    # samples.json 로드
    with open(args.samples, encoding="utf-8") as f:
        manifest = json.load(f)

    samples: list[dict[str, Any]] = manifest.get("samples", [])
    if not samples:
        logger.error("samples.json 이 비어있습니다.")
        return 1

    # 모델 로드 (TESTING 플래그 강제 해제 + 최초 로딩은 측정 대상 외)
    os.environ.pop("TESTING", None)
    from app import stt_processor  # noqa: E402 — 경로 설정 후 import

    model_size = os.environ.get("MODEL_SIZE", "(config default)")
    logger.info("라벨: %s | MODEL_SIZE=%s", args.label, model_size)
    logger.info("샘플 수: %d", len(samples))

    logger.info("모델 로드 시작...")
    t_load_start = time.time()
    stt_processor.load_models()
    load_elapsed = time.time() - t_load_start
    logger.info("모델 로드 완료 (%.1fs)", load_elapsed)

    # 로드 직후 VRAM 기록
    try:
        vram_after_load = int(
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        vram_after_load = -1
    logger.info("로드 직후 VRAM: %d MiB", vram_after_load)

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples, 1):
        hash_id = sample["hash"]
        rel_file = sample["file"]
        duration = sample.get("duration", 0)
        audio_path = args.audio_root / rel_file
        if not audio_path.exists():
            logger.error("[%d/%d] %s 파일 없음: %s", i, len(samples), hash_id, audio_path)
            continue

        logger.info("[%d/%d] %s | %ds | %s", i, len(samples), hash_id, duration, audio_path.name)

        # stt_processor.transcribe()는 입력 파일을 삭제하므로 tmp 복사본 전달
        tmp_file = Path(tempfile.mkdtemp(prefix="bench-")) / audio_path.name
        shutil.copy2(audio_path, tmp_file)

        sampler = VramSampler(interval_sec=0.5)
        sampler.start()

        t0 = time.time()
        error: str | None = None
        try:
            result = stt_processor.transcribe(
                file_path=str(tmp_file),
                task_id=f"bench-{args.label}-{hash_id}",
                enable_diarize=(not args.skip_diarize),
                enable_name_masking=True,
                mask_pii=True,
                split_by_speaker=False,
                split_by_utterance=True,
            )
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception("[%d/%d] %s 실패", i, len(samples), hash_id)
            result = {}
        elapsed = time.time() - t0

        sampler.stop()
        sampler.join(timeout=2)
        peak_vram = max(sampler.samples) if sampler.samples else -1

        # 임시 파일 정리 (stt_processor가 원본을 지웠을 수도, 안 지웠을 수도)
        try:
            shutil.rmtree(tmp_file.parent, ignore_errors=True)
        except Exception:
            pass

        # 전체 결과 저장 (오디오 바이트는 재귀적으로 제거)
        clean_result = _strip_bytes(result)
        result_path = args.out / f"{hash_id}.result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)

        # 세그먼트 텍스트 연결 (WER 계산용)
        segments = clean_result.get("segments", []) or clean_result.get("utterances", [])
        transcript_text = " ".join(
            (s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip()
        )

        rtf = elapsed / duration if duration > 0 else -1.0
        meta = {
            "hash": hash_id,
            "label": args.label,
            "model_size": model_size,
            "duration_sec": duration,
            "elapsed_sec": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "peak_vram_mib": peak_vram,
            "vram_after_load_mib": vram_after_load,
            "segment_count": len(segments),
            "transcript_len_chars": len(transcript_text),
            "error": error,
        }
        meta_path = args.out / f"{hash_id}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            "[%d/%d] %s 완료 | elapsed=%.1fs | rtf=%.3f | peak=%d MiB | segs=%d | chars=%d%s",
            i, len(samples), hash_id, elapsed, rtf, peak_vram,
            len(segments), len(transcript_text),
            f" | ERROR: {error}" if error else "",
        )
        results.append(meta)

    summary = {
        "label": args.label,
        "model_size": model_size,
        "load_elapsed_sec": round(load_elapsed, 2),
        "vram_after_load_mib": vram_after_load,
        "sample_count": len(results),
        "avg_rtf": round(
            sum(r["rtf"] for r in results if r["rtf"] > 0) / max(1, sum(1 for r in results if r["rtf"] > 0)),
            3,
        ),
        "peak_vram_mib_overall": max((r["peak_vram_mib"] for r in results), default=-1),
        "results": results,
    }
    summary_path = args.out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("요약 저장: %s", summary_path)
    logger.info("평균 RTF: %.3f | 피크 VRAM: %d MiB", summary["avg_rtf"], summary["peak_vram_mib_overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
