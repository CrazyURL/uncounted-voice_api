# Voice API Reference

Base URL: `http://{host}:{port}/api/v1`
Swagger UI: `http://{host}:{port}/docs`

---

## GET /health

서버 상태 및 GPU/모델 정보 확인.

**Response 200**
```json
{
  "status": "ok",
  "service": "WhisperX STT Server",
  "version": "2.0.0",
  "model": "large-v3",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "model_loaded": true
}
```

`model_loaded: false`이면 모델 로딩 중 — STT 요청 보내면 안 됨.

---

## POST /transcribe

음성 파일 업로드 → 비동기 STT 작업 생성.

**Request**: `multipart/form-data`

| 파라미터 | 위치 | 타입 | 기본 | 설명 |
|----------|------|------|------|------|
| `file` | body | file | 필수 | 음성 파일 (wav, mp3, m4a, ogg, flac, webm, mp4) |
| `language` | query | string | `ko` | 인식 언어 (ISO 639-1) |
| `diarize` | query | bool | `false` | 화자분리 활성화 (HF_TOKEN 필요) |
| `mask_pii` | query | bool | `true` | PII 자동 마스킹 |
| `enable_name_masking` | query | bool | `false` | 한국어 이름 마스킹 |
| `split_by_speaker` | query | bool | `false` | 화자별 오디오 분리 (diarize=true 필수) |
| `split_by_utterance` | query | bool | `false` | 발화별 오디오 분리 (diarize=true 필수) |

**Response 200**
```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "pending"
}
```

**Errors**: 400 (지원하지 않는 포맷), 413 (500MB 초과)

**curl 예시**
```bash
curl -X POST "http://host:8000/api/v1/transcribe?diarize=true&split_by_utterance=true&split_by_speaker=true" \
  -F "file=@audio.m4a"
```

---

## GET /jobs/{task_id}

작업 상태 및 결과 조회. 폴링 간격 1~2초 권장.

**Response 200 (processing)**
```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "processing"
}
```

**Response 200 (completed — 전체 옵션 적용)**
```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "completed",
  "language": "ko",
  "duration_seconds": 20.17,
  "diarization_enabled": true,
  "segments": [
    {
      "start": 0.03,
      "end": 1.19,
      "text": "왜 그러지?",
      "speaker": "SPEAKER_02",
      "words": [
        {"word": "왜", "start": 0.03, "end": 0.59, "speaker": "SPEAKER_02"},
        {"word": "그러지?", "start": 0.73, "end": 1.19, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "full_text": "왜 그러지? 어 끊어졌네 갑자기 ...",
  "pii_summary": [{"type": "이름", "count": 1}],
  "utterances": [
    {
      "index": 0,
      "start_sec": 0.03,
      "end_sec": 2.79,
      "duration_sec": 2.76,
      "speaker_id": "SPEAKER_02",
      "transcript_text": "왜 그러지? 어 끊어졌네 갑자기",
      "audio_filename": "utterance_000.wav",
      "words": [
        {"word": "왜", "start": 0.03, "end": 0.59, "speaker": "SPEAKER_02"},
        {"word": "그러지?", "start": 0.73, "end": 1.19, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "speaker_audio": [
    {
      "speaker_id": "SPEAKER_01",
      "total_duration_sec": 246.91,
      "audio_filename": "speaker_speaker_01.wav"
    },
    {
      "speaker_id": "SPEAKER_02",
      "total_duration_sec": 246.91,
      "audio_filename": "speaker_speaker_02.wav"
    }
  ]
}
```

**필드 설명**

| 필드 | 조건 | 설명 |
|------|------|------|
| `segments` | 항상 | WhisperX 세그먼트 (타임스탬프 + 텍스트 + 화자) |
| `segments[].words` | diarize=true | word 단위 타임스탬프 + 화자 |
| `utterances` | split_by_utterance=true | 발화 분리 결과 (WAV 1:1 매핑) |
| `utterances[].words` | split_by_utterance=true | 발화 내 word 단위 데이터 |
| `speaker_audio` | split_by_speaker=true | 화자별 mute 오디오 (원본 타임라인 유지, 상대방 무음) |
| `pii_summary` | mask_pii=true | 마스킹된 PII 유형별 건수 |

**Errors**: 404 (task 없음), 500 (처리 실패)

---

## GET /jobs/{task_id}/audio/{filename}

분리된 WAV 오디오 바이너리 다운로드. 인메모리에서 직접 서빙.

**파일명 규칙**
- 발화별: `utterance_000.wav`, `utterance_001.wav`, ...
- 화자별: `speaker_speaker_01.wav`, `speaker_speaker_02.wav`, ...

**Response 200**: `Content-Type: audio/wav` (바이너리)

**curl 예시**
```bash
curl -o utterance_000.wav "http://host:8000/api/v1/jobs/{task_id}/audio/utterance_000.wav"
```

**Errors**: 404 (task 또는 파일 없음)

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ENV` | `dev` | 환경 (dev/live) |
| `PORT` | `8001`(dev) / `8000`(live) | 서버 포트 |
| `MODEL_SIZE` | `large-v3` | WhisperX 모델 |
| `DEVICE` | `cuda` | 추론 디바이스 |
| `COMPUTE_TYPE` | `float16` | 연산 정밀도 |
| `LANGUAGE` | `ko` | 인식 언어 |
| `BATCH_SIZE` | `4` | 배치 크기 |
| `HF_TOKEN` | - | HuggingFace 토큰 (화자분리용) |
| `HOTWORDS` | - | 고유명사 힌트 (쉼표 구분, 따옴표 필수) |
| `INITIAL_PROMPT` | - | STT 초기 프롬프트 (따옴표 필수) |
| `TEMP_DIR` | `/dev/shm/stt-temp` | 임시 파일 경로 |
| `MAX_UPLOAD_SIZE` | `524288000` | 업로드 제한 (500MB) |

---

## 처리 파이프라인

```
POST /transcribe (파일 업로드)
  └─ BackgroundTask
       ├─ [1] whisperx.load_audio + model.transcribe     → STT
       ├─ [2] whisperx.align                              → 타임스탬프 정렬
       ├─ [3] pyannote 화자분리 (diarize=true)            → word별 speaker
       ├─ [4] 세그먼트 정리 (word 데이터 보존)
       ├─ [5] PII 마스킹 (mask_pii=true)
       ├─ [6] null speaker 보정 (인접 화자 상속)
       ├─ [7] 발화 분리 (split_by_utterance=true)
       │    ├─ 화자변경 + 0.5초 묵음 → 경계
       │    ├─ 5초 미만 → 병합 (단답형 보존)
       │    ├─ 30초 초과 → 문장경계 분할
       │    └─ 1초 미만 초단발화 → 맞장구 병합
       ├─ [8] 화자별 오디오 (split_by_speaker=true)
       │    └─ mute 방식 (원본 타임라인 + 상대방 무음)
       ├─ [9] WAV bytes → job_store._audio (인메모리)
       └─ [10] 원본 삭제 + VRAM 정리
```
