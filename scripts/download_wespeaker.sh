#!/bin/bash
# Download WeSpeaker ResNet34-LM ONNX model for Phase 4 speaker embedding extraction.
# Usage: ./download_wespeaker.sh [model-dir]
# Default model-dir: ~/.cache/uncounted/wespeaker/

set -e

MODEL_DIR="${1:- $HOME/.cache/uncounted/wespeaker}"
ONNX_FILE="$MODEL_DIR/wespeaker_resnet34lm.onnx"
HF_REPO="wenet-speaker/wespeaker_resnet34lm_zh"
HF_FILE="pytorch_model.bin"

echo "WeSpeaker ONNX Downloader"
echo "Model directory: $MODEL_DIR"
echo "Target ONNX file: $ONNX_FILE"

# Check if file already exists
if [ -f "$ONNX_FILE" ]; then
  echo "✓ Model already downloaded: $ONNX_FILE"
  exit 0
fi

# Create directory if needed
mkdir -p "$MODEL_DIR"

# Check for huggingface-hub CLI
if ! command -v huggingface-cli &> /dev/null; then
  echo "Error: huggingface-hub is required. Install with: pip install huggingface-hub"
  exit 1
fi

echo "Downloading model from HuggingFace: $HF_REPO/$HF_FILE ..."

# Download using huggingface-cli
huggingface-cli download "$HF_REPO" "$HF_FILE" --cache-dir "$MODEL_DIR" --local-dir "$MODEL_DIR"

if [ -f "$MODEL_DIR/$HF_FILE" ]; then
  echo "✗ Downloaded PyTorch model (.bin), but Phase 4 requires ONNX (.onnx)"
  echo ""
  echo "To convert PyTorch to ONNX:"
  echo "  1. Clone WeSpeaker repo: git clone https://github.com/wenet-e2e/wespeaker"
  echo "  2. Run: python scripts/export_onnx.py --model-path pytorch_model.bin --output-file wespeaker_resnet34lm.onnx"
  echo ""
  echo "For pre-converted ONNX, check: https://github.com/wenet-e2e/wespeaker/releases"
  exit 1
fi

echo "✓ Model download complete: $ONNX_FILE"
