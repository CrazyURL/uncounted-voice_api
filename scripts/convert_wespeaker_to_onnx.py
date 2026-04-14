#!/usr/bin/env python
"""pyannote/wespeaker-voxceleb-resnet34-LM PyTorch → ONNX 변환.

사용법:
    cd uncounted-voice-api
    source venv/bin/activate
    python scripts/convert_wespeaker_to_onnx.py

전제:
    HF에서 pyannote/wespeaker-voxceleb-resnet34-LM 모델이 이미 다운로드되어
    있어야 합니다 (HF_HUB_CACHE 또는 ~/.cache/huggingface). HF_TOKEN 필요.

산출:
    models/wespeaker_resnet34_LM.onnx

출력 절대경로를 voice-api .env.dev 의 VOICE_DIARIZATION_WESPEAKER_MODEL_PATH
에 설정한 후 systemctl restart voice-api@dev.
"""

import os
import sys
from pathlib import Path

import torch

try:
    from pyannote.audio import Model
except ImportError as e:
    print(f"ERROR: pyannote.audio import 실패: {e}")
    print("voice-api venv를 활성화했는지 확인하세요: source venv/bin/activate")
    sys.exit(1)


REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"
OUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUT_PATH = OUT_DIR / "wespeaker_resnet34_LM.onnx"
SAMPLE_RATE = 16_000
SAMPLE_SECONDS = 2
OPSET_VERSION = 14


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading: {REPO}")
    use_auth_token = os.environ.get("HF_TOKEN")
    model = Model.from_pretrained(REPO, use_auth_token=use_auth_token)
    model.eval()

    # WeSpeaker 입력: shape (batch, channels, samples), 16kHz mono
    dummy = torch.randn(1, 1, SAMPLE_RATE * SAMPLE_SECONDS)

    print(f"sanity forward pass: input {tuple(dummy.shape)}")
    with torch.no_grad():
        sample_out = model(dummy)
    print(f"  output shape: {tuple(sample_out.shape)}")

    print(f"exporting ONNX → {OUT_PATH}")
    # pyannote/wespeaker는 forward 내부에서 torch.vmap(kaldi.fbank)을 호출한다.
    # 레거시 TorchScript tracing은 vmap을 지원하지 않으므로 새 dynamo exporter를 사용.
    # dynamo=True는 PyTorch 2.5+ 와 onnxscript 패키지가 필요하다.
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT_PATH),
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "embedding": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        dynamo=True,
    )

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"done: {OUT_PATH} ({size_mb:.1f} MB)")
    print()
    print(f"absolute path:  {OUT_PATH}")
    print()
    print("Next:")
    print(f"  echo 'VOICE_DIARIZATION_WESPEAKER_MODEL_PATH={OUT_PATH}' >> .env.dev")
    print("  sudo systemctl restart voice-api@dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
