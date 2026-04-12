#!/bin/bash
set -e
cd "$(dirname "$0")"

ENV=${1:-dev}
ENV_FILE=".env.${ENV}"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Usage: ./run.sh [dev|live]"
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

source venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$TEMP_DIR" "$RESULTS_DIR"

echo "=========================================="
echo " WhisperX STT Server ($ENV)"
echo "=========================================="
echo "Port: $PORT"
echo "Device: $DEVICE"
echo "Model: $MODEL_SIZE"
echo "Log Level: $LOG_LEVEL"
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN: set (diarization enabled)"
else
    echo "HF_TOKEN: not set (diarization disabled)"
fi
echo ""

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level "$(echo $LOG_LEVEL | tr '[:upper:]' '[:lower:]')"
