#!/bin/bash
set -e

echo '=========================================='
echo ' STT Server 설치 스크립트'
echo '=========================================='

# 시스템 패키지
echo '[1/4] 시스템 패키지 설치...'
sudo apt update && sudo apt install -y python3-venv ffmpeg

# 가상환경
echo '[2/4] Python 가상환경 생성...'
cd ~/project/Uncounted-root/uncounted-voice-api
python3 -m venv venv
source venv/bin/activate

# PyTorch CUDA 12.4
echo '[3/4] PyTorch CUDA 12.4 설치...'
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# WhisperX + FastAPI
echo '[4/4] WhisperX 및 서버 패키지 설치...'
pip install git+https://github.com/m-bain/whisperx.git
pip install fastapi "uvicorn[standard]" python-multipart aiofiles

echo ''
echo '=========================================='
echo ' 설치 완료!'
echo '=========================================='

# GPU 확인
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('WARNING: CUDA not available')"
