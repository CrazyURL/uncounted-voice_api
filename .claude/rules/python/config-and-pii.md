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
| `MODEL_SIZE` | `large-v3` | WhisperX model size |
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

### Audio Preprocessing

| Variable | Default | Notes |
|----------|---------|-------|
| `DENOISE_ENABLED` | `true` | DeepFilterNet background noise removal |
| `SILENCE_RMS_THRESHOLD` | `0.01` | RMS threshold for silence detection |
| `DUPLICATE_WINDOW_SEC` | `2.5` | Dedup sliding window size (seconds) |
| `DUPLICATE_CORR_THRESHOLD` | `0.85` | Cross-correlation threshold for duplicate detection |
| `PREPROCESS_FRAME_MS` | `20` | Frame size for silence compression (ms) |
| `MAX_DEDUP_LOOKAHEAD` | `5` | Max forward windows to compare for dedup |
| `LOCAL_MAX_GAIN_X` | `30.0` | 500ms 슬라이딩 윈도우 로컬 게인 최대 배수 |

### VAD & Diarization

| Variable | Default | Notes |
|----------|---------|-------|
| `VAD_ONSET` | `0.150` | Silero VAD 활성화 임계값 (낮을수록 민감) |
| `VAD_OFFSET` | `0.100` | Silero VAD 비활성화 임계값 |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | 화자분리 모델 ID |
| `HANGING_WORD_GAP_SEC` | `0.3` | 발화 끝 단어 gap 기준 (초) — 이 이상 gap 후 고립 단어는 다음 발화로 이동 |

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
