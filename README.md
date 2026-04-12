# uncounted-voice-api

WhisperX 기반 음성 처리 API 서버. STT, 화자 분리, PII 비식별화를 처리한다.

## 기술 스택

- Python 3.12 + FastAPI
- WhisperX (large-v3, CUDA)
- pyannote/speaker-diarization (HF_TOKEN 필요)
- PII 마스킹 (한국어 특화, regex 기반)

## 환경 설정

| 파일 | 환경 | 포트 | 로그 레벨 |
|------|------|------|----------|
| .env.dev | 개발 | 8001 | DEBUG |
| .env.live | 운영 | 8000 | INFO |

HF_TOKEN을 .env 파일에 추가하면 화자분리가 활성화된다.

## 실행

### 직접 실행



### systemd 서비스



### Cloudflare Tunnel (외부 공개)



## API 엔드포인트

### POST /api/v1/transcribe

음성 파일을 업로드하여 STT 처리를 시작한다.

    curl -X POST http://localhost:8000/api/v1/transcribe -F "file=@audio.wav" -F "diarize=true"

응답: { "task_id": "abc123", "status": "pending" }

### GET /api/v1/jobs/{task_id}

처리 결과를 조회한다.

    curl http://localhost:8000/api/v1/jobs/abc123

### GET /api/v1/health

서버 상태를 확인한다.

    curl http://localhost:8000/api/v1/health

## 테스트

    source venv/bin/activate
    python -m pytest tests/ -v

## Docker

    docker compose up -d

GPU 사용을 위해 nvidia-container-toolkit이 필요하다.

## 빠른 참조

| 명령 | 설명 |
|------|------|
| sudo systemctl start voice-api@live | live 서버 시작 (port 8000) |
| sudo systemctl start voice-api@dev | dev 서버 시작 (port 8001) |
| sudo systemctl start cloudflared-voice | Cloudflare 터널 시작 |
| sudo systemctl stop voice-api@live | live 서버 중지 |
| journalctl -fu voice-api@live | live 로그 실시간 |
