---
paths:
  - "app/stt_processor.py"
  - "app/services/whisperx_service.py"
  - "app/services/audio_preprocessor.py"
  - "app/services/utterance_segmenter.py"
  - "app/services/audio_splitter.py"
  - "app/routers/transcribe.py"
---

# STT Pipeline Architecture

## Processing Flow

```
Upload → Preprocess (gain → denoise → dedup → silence) → GPU Inference → Alignment → (Diarization) → PII Masking → Result
```

## Key Components

| Module | Role |
|--------|------|
| `stt_processor.py` | Core pipeline orchestrator. GPU lock, chunked mode, model lifecycle |
| `audio_preprocessor.py` | 4-stage preprocessing: gain normalize → DeepFilterNet denoise → duplicate removal → silence compression |
| `whisperx_service.py` | Thin wrapper (static methods) over stt_processor |
| `utterance_segmenter.py` | Word-level utterance boundary detection |
| `audio_splitter.py` | Per-utterance/per-speaker WAV extraction |
| `pii_masker.py` | Korean PII regex masking (9 types + optional name masking) |

## GPU Singleton Pattern

- All models (`_model`, `_align_model`, `_diarize_model`) are global singletons loaded once at startup via `lifespan`
- `_gpu_lock = threading.Semaphore(1)` — only 1 concurrent GPU inference
- `torch.cuda.empty_cache()` called inside the lock's `finally` block
- Single worker (`--workers 1`) is mandatory — GPU models are not fork-safe

## Chunked Mode (Large Audio)

Files longer than `CHUNK_THRESHOLD_SEC` (default 1h) trigger chunked processing:

1. `ffmpeg silencedetect` finds silence points (no memory usage)
2. Split points chosen near target chunk length (`CHUNK_DURATION_SEC`, default 30min)
3. Each chunk: `ffmpeg` extract → load → preprocess → GPU inference → offset timestamps
4. Chunk WAV deleted immediately after processing
5. Speaker/utterance audio splitting disabled in chunked mode (no full audio array)

## Async Job Pattern

- `POST /transcribe` returns `task_id` immediately (202-style via 200)
- `BackgroundTasks` runs `_process_audio` in background
- `JobStore` (in-memory, thread-safe) tracks status: pending → processing → completed/failed
- Client polls `GET /jobs/{task_id}` (recommended 1-2s interval)
- Completed jobs auto-expire after 1h, max 100 stored

## Queue Backpressure (503)

- `JobStore.active_count()`이 `MAX_ACTIVE_JOBS`(기본 5) 이상이면 `POST /transcribe`는 503 반환
- 응답에 `Retry-After: 30` 헤더 + JSON body `detail.retry_after_sec: 30` 동시 포함
- 목적: 긴 세션이 GPU Semaphore 점유 중일 때 짧은 세션이 무한 큐잉되어 클라이언트 폴링 타임아웃 루프에 빠지는 것을 방지
- `GET /health` 응답에 `queue: {active, max_active, utilization_pct}` 필드로 현재 큐 포화도 관측 가능
- 관련 이슈 문서: `uncounted-docs/voice-api/큐_병목_및_폴링_타임아웃_이슈.md`

## Test Environment

- `TESTING=1` 환경변수가 설정되면 lifespan이 `whisperx_service.load_models()`를 스킵
- 이유: 운영 서버가 이미 GPU를 점유 중일 때 pytest의 TestClient가 lifespan으로 모델을 재로딩 시도 → CUDA OOM 방지
- `tests/conftest.py`가 import 시점에 `os.environ.setdefault("TESTING", "1")` 자동 설정

## Audio File Storage

- Upload temp: `/dev/shm/stt-temp` (RAM disk)
- Result WAVs: `/dev/shm/stt-results/{task_id}/` (RAM disk)
- Fallback to `/tmp/` if `/dev/shm` unavailable
- Original audio deleted after processing (in `finally` block)
- Path traversal protection on audio download endpoint
