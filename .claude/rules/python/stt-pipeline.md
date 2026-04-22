---
paths:
  - "app/stt_processor.py"
  - "app/services/whisperx_service.py"
  - "app/services/audio_preprocessor.py"
  - "app/services/utterance_segmenter.py"
  - "app/services/audio_splitter.py"
  - "app/services/audio_pii_masker.py"
  - "app/routers/transcribe.py"
---

# STT Pipeline Architecture

## Processing Flow

```
Upload → Preprocess (gain → denoise → dedup → silence) → GPU Inference → Alignment →
  (Diarization) → [Audio PII Masking, opt-in] → Text PII Masking → Speaker/Utterance Split → Result
```

- **Audio PII Masking**은 `mask_audio_pii=true`일 때 음성(PCM)의 PII 구간을 1kHz 비프로 치환. Text masking 전에 수행되어 생성되는 `speaker_audio`/`utterance_audio`에 자연히 반영된다.
- **Text PII Masking**은 segment text의 regex 치환. 항상 수행(`mask_pii=true` 기본).

## Key Components

| Module | Role |
|--------|------|
| `stt_processor.py` | Core pipeline orchestrator. GPU lock, chunked mode, model lifecycle |
| `audio_preprocessor.py` | 4-stage preprocessing: gain normalize → DeepFilterNet denoise → duplicate removal → silence compression |
| `whisperx_service.py` | Thin wrapper (static methods) over stt_processor |
| `utterance_segmenter.py` | Word-level utterance boundary detection |
| `audio_splitter.py` | Per-utterance/per-speaker WAV extraction |
| `pii_masker.py` | Korean PII regex: `detect_pii_spans` (위치), `mask_pii` (치환), 9 types + optional name masking |
| `audio_pii_masker.py` | Audio-level PII: char span→word timestamp 매핑 + 1kHz 비프 치환 (10ms fade) |

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

## Audio File Storage

- Upload temp: `/dev/shm/stt-temp` (RAM disk)
- Result WAVs: `/dev/shm/stt-results/{task_id}/` (RAM disk)
- Fallback to `/tmp/` if `/dev/shm` unavailable
- Original audio deleted after processing (in `finally` block)
- Path traversal protection on audio download endpoint
