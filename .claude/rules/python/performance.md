---
paths:
  - "app/stt_processor.py"
  - "app/services/audio_preprocessor.py"
  - "app/core/job_store.py"
---

# Performance Optimizations

## DeepFilterNet CUDA Isolation

DeepFilterNet runs as a persistent daemon subprocess (CPU-only) to avoid CUDA allocator conflicts with WhisperX:

- Separate Python process with `DF_DEVICE=cpu`
- `__globals__` monkey-patch forces `get_device()` → CPU in all df.enhance functions
- File-based IPC: `input.raw` → `request` signal → worker processes → `output.raw` → `done` signal
- Auto-restart on worker death or work_dir deletion
- Timeout: `max(60s, audio_duration * 2 + 30s)`

## Preprocessing Pipeline (4 stages)

1. **Gain normalize** — RMS-based, only boosts low signals (TARGET_RMS=0.1, MAX_GAIN=20x)
2. **Denoise** — DeepFilterNet daemon (optional, `denoise_enabled` flag)
3. **Deduplicate** — Sliding window cross-correlation, O(n*K) where K=`MAX_DEDUP_LOOKAHEAD` (default 5)
4. **Compress silence** — RMS energy per 20ms frame, silence > 0.5s compressed to 0.5s

## Memory Management

- Streaming upload: 64KB chunks with size tracking (no full file in memory)
- RAM disk (`/dev/shm`) for temp files and results — avoids disk I/O
- Chunked processing for audio > 1h: chunk WAV deleted immediately after GPU inference
- `torch.cuda.empty_cache()` inside GPU lock's `finally` block
- Startup cleanup: removes stale temp files from previous sessions

## Chunked Mode Response Contract

원본 길이 ≥ `CHUNK_THRESHOLD_SEC`(기본 1h) 파일은 청크 모드로 처리되며, 응답에서 다음과 같이 동작한다:

- `chunked_processing: true`
- `utterances`: **청크 모드에서도 제공**. 각 청크가 메모리에 있는 동안 chunk-local 좌표로 발화 경계를 산출한 뒤 `extract_utterance_audio_local`로 WAV를 즉시 생성하고, 메타데이터 타임스탬프만 누적 offset으로 globalize한다. 청크 경계에 걸치는 발화는 경계에서 강제 분리될 수 있다.
- `speaker_audio`: **청크 모드에서는 `null` 고정**. 화자별 WAV는 전체 오디오 배열이 필요해 OOM 제약 때문에 제공하지 않는다.
- `_audio_files`: `utterance_XXX.wav`만 포함 (speaker_XX.wav 없음)

관련 이슈 문서: `uncounted-docs/voice-api/청크모드_WAV_미생성_이슈.md`

## Critical Constraints

- **Single worker only** (`--workers 1`): GPU models are global singletons, not fork-safe
- **GPU Semaphore(1)**: Never allow concurrent GPU inference
- **BATCH_SIZE=2** (reduced from 4): Lower VRAM usage for large-v3 model
- **In-memory job store**: All state lost on server restart (max 100 jobs, 1h TTL)
- **ffmpeg required**: System dependency for silence detection and chunk extraction
