"""Turbo 모델 로드 + VRAM 측정 (STT 실행 없음).

사용법 (WSL):
    MODEL_SIZE=large-v3-turbo python scripts/turbo_load_probe.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    os.environ.pop("TESTING", None)
    from app import stt_processor

    model_size = os.environ.get("MODEL_SIZE", "(config default)")
    print(f"MODEL_SIZE={model_size}")

    t0 = time.time()
    stt_processor.load_models()
    elapsed = time.time() - t0
    print(f"load_elapsed: {elapsed:.1f}s")

    vram = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    print(f"vram_after_load: {vram}")


if __name__ == "__main__":
    main()
