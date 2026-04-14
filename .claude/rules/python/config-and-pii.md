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

`ALLOWED_EXTENSIONS` (config.py 상수): `wav, mp3, m4a, ogg, flac, webm, mp4, amr, 3gp`. Android 통화 녹음을 위해 `amr`, `3gp` 추가 (2026-04-14, `57c2747`). `3gp`는 ffmpeg 통합 demuxer(`mov,mp4,m4a,3gp,3g2,mj2`)로 디코드된다. 확장자 거절 시 warning 로그는 `ext`와 `filename_len`만 기록 — 파일명은 PII (전화번호/인명) 포함 가능하므로 평문 로깅 금지.

### Speaker Diarization (Option D + Option B)

Option D — production 적용 (pyannote `min_speakers=2, max_speakers=2` 힌트):

| Variable | Default | Notes |
|----------|---------|-------|
| `VOICE_DIARIZATION_FORCE_TWO_SPEAKERS` | `false` | true 시 pyannote에 force_two_speakers 힌트. dev는 현재 `true`, live는 환경별 확인 |
| `VOICE_DIARIZATION_ENDPOINT_MODES` | `call_recording` | Option D 적용 엔드포인트 모드 allowlist |

Option B — WeSpeaker 재클러스터링 (검증 보류, production flag-off 유지):

| Variable | Default | Notes |
|----------|---------|-------|
| `VOICE_DIARIZATION_WESPEAKER_RECLUSTER` | `false` | Phase 7 훅 활성화 여부 |
| `VOICE_DIARIZATION_WESPEAKER_RECLUSTER_ENDPOINTS` | `call_recording` | 재클러스터링 활성 엔드포인트 |
| `VOICE_DIARIZATION_WESPEAKER_REPO` | `pyannote/wespeaker-voxceleb-resnet34-LM` | pyannote.audio.Model.from_pretrained repo (HF_TOKEN 필요) |
| `VOICE_DIARIZATION_EMBEDDING_PROVIDER` | `cpu` | `cpu` / `cuda`. `cuda` 선택 시 사용 불가면 cpu 폴백 |
| `VOICE_DIARIZATION_RECLUSTER_CONFIDENCE_THRESHOLD` | `0.30` | AHC cosine margin 미만 시 원본 라벨 유지 (2026-04-15: 0.15 → 0.30 상향, real-audio 검증 반영) |
| `VOICE_DIARIZATION_RECLUSTER_MIN_WINDOW_SEC` | `1.0` | 임베딩 윈도우 최소 길이 |
| `VOICE_DIARIZATION_RECLUSTER_MAX_WINDOW_SEC` | `4.0` | 임베딩 윈도우 최대 길이 |

관련 이슈 문서: `uncounted-docs/voice-api/Option_B_WeSpeaker_재클러스터링_검증_보류.md`, `uncounted-docs/voice-api/화자분리_정확도_이슈.md`

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
