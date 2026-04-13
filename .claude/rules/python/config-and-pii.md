# Configuration & PII Masking

## Environment Variables

### Core

| Variable | Default | Notes |
|----------|---------|-------|
| `ENV` | `dev` | Environment (`dev` / `live`) |
| `PORT` | `8001`(dev) / `8000`(live) | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `WORKERS` | `1` | Uvicorn workers (must be 1 — GPU singleton) |
| `LOG_LEVEL` | `DEBUG`(dev) / `INFO`(live) | Logging level |

### WhisperX Model

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_SIZE` | `large-v3` (config.py 기본값) | dev 환경은 `.env.dev`로 `large-v3-turbo` 오버라이드 (2026-04-13 전환). live는 large-v3 유지 |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` | Model precision |
| `LANGUAGE` | `ko` | Target language |
| `BATCH_SIZE` | `2` | Transcription batch size (reduced for VRAM) |
| `HF_TOKEN` | (none) | HuggingFace token; required for speaker diarization |
| `HOTWORDS` | (none) | Proper noun hints (comma-separated, quotes required) |
| `INITIAL_PROMPT` | (none) | STT initial prompt (quotes required) |

### Storage & Upload

| Variable | Default | Notes |
|----------|---------|-------|
| `TEMP_DIR` | `/dev/shm/stt-temp` | Upload temp storage (RAM disk) |
| `RESULTS_DIR` | `/dev/shm/stt-results` | Result WAV storage |
| `MAX_UPLOAD_SIZE` | `524288000` | Upload size limit (500MB) |

### Queue Backpressure

| Variable | Default | Notes |
|----------|---------|-------|
| `MAX_ACTIVE_JOBS` | `5` | pending + processing 합산이 이 값에 도달하면 `POST /transcribe`는 503 반환 |
| `QUEUE_FULL_RETRY_AFTER_SEC` | `30` | 503 응답의 `Retry-After` 헤더 + JSON body `detail.retry_after_sec` 값 |
| `TESTING` | (unset) | `1`로 설정 시 lifespan에서 WhisperX 모델 로딩을 스킵 (pytest 전용) |

### Audio Preprocessing

| Variable | Default | Notes |
|----------|---------|-------|
| `DENOISE_ENABLED` | `true` | DeepFilterNet background noise removal |
| `SILENCE_RMS_THRESHOLD` | `0.01` | RMS threshold for silence detection |
| `DUPLICATE_WINDOW_SEC` | `2.5` | Dedup sliding window size (seconds) |
| `DUPLICATE_CORR_THRESHOLD` | `0.85` | Cross-correlation threshold for duplicate detection |
| `PREPROCESS_FRAME_MS` | `20` | Frame size for silence compression (ms) |
| `MAX_DEDUP_LOOKAHEAD` | `5` | Max forward windows to compare for dedup |

### Large Audio Chunking

| Variable | Default | Notes |
|----------|---------|-------|
| `CHUNK_THRESHOLD_SEC` | `3600` | Trigger chunked mode above this duration (1h) |
| `CHUNK_DURATION_SEC` | `1800` | Target chunk length (30min) |
| `CHUNK_SILENCE_DB` | `-30` | ffmpeg silencedetect threshold (dB) |
| `CHUNK_SILENCE_DUR` | `0.3` | Minimum silence duration for split point (seconds) |
| `CHUNK_MARGIN_SEC` | `300` | Split point search range around target (±5min) |

## PII Masking Details

`pii_masker.py` handles Korean-specific PII: resident registration numbers, driver's license, passport, card numbers, email, phone, bank accounts, IP addresses. Korean name masking is opt-in (`enable_name_masking`) and uses a surname list + context heuristics with an extensive exclude-prefix set to avoid false positives on common Korean words that start with surname characters.
