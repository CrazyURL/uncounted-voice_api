"""화자분리 모델 VRAM 사용량 비교 스크립트.

community-1 vs speaker-diarization-3.1 의 VRAM 차이를 측정한다.

사용법:
    HF_TOKEN=<token> python scripts/check_diarize_vram.py
"""

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch


def _vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def _vram_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 1024 / 1024


def _free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _load_model(model_name: str, hf_token: str):
    from whisperx.diarize import DiarizationPipeline
    return DiarizationPipeline(use_auth_token=hf_token, device="cuda", model_name=model_name)


def _measure(model_name: str, hf_token: str) -> dict:
    _free_cuda()
    before = _vram_mb()
    before_reserved = _vram_reserved_mb()

    print(f"\n  [{model_name}] 로딩 중...", flush=True)
    try:
        model = _load_model(model_name, hf_token)
        after = _vram_mb()
        after_reserved = _vram_reserved_mb()
        delta = after - before
        delta_reserved = after_reserved - before_reserved
        print(f"  [{model_name}] 로딩 완료", flush=True)
    except Exception as e:
        print(f"  [{model_name}] 로딩 실패: {e}", flush=True)
        return {"model": model_name, "error": str(e)}
    finally:
        try:
            del model
        except Exception:
            pass
        _free_cuda()

    return {
        "model": model_name,
        "before_mb": round(before, 1),
        "after_mb": round(after, 1),
        "delta_allocated_mb": round(delta, 1),
        "delta_reserved_mb": round(delta_reserved, 1),
    }


def main():
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[ERROR] HF_TOKEN 환경변수가 없습니다.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 없음 — GPU 서버에서 실행하세요.", file=sys.stderr)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    print(f"\nGPU: {device_name}")
    print(f"총 VRAM: {total_vram:.0f} MB ({total_vram/1024:.1f} GB)")

    models = [
        "pyannote/speaker-diarization-community-1",
        "pyannote/speaker-diarization-3.1",
    ]

    results = []
    for m in models:
        r = _measure(m, hf_token)
        results.append(r)

    print("\n" + "=" * 60)
    print("  VRAM 사용량 비교")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<45}  ERROR: {r['error']}")
        else:
            print(
                f"  {r['model']:<45}  "
                f"+{r['delta_allocated_mb']:>6.0f} MB (allocated)  "
                f"+{r['delta_reserved_mb']:>6.0f} MB (reserved)"
            )

    if len(results) == 2 and "error" not in results[0] and "error" not in results[1]:
        diff = results[1]["delta_allocated_mb"] - results[0]["delta_allocated_mb"]
        diff_r = results[1]["delta_reserved_mb"] - results[0]["delta_reserved_mb"]
        sign = "+" if diff >= 0 else ""
        print(f"\n  3.1 - community-1 차이: {sign}{diff:.0f} MB (allocated)  {sign}{diff_r:.0f} MB (reserved)")

    print("=" * 60)


if __name__ == "__main__":
    main()
