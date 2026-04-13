# uncounted-voice-api — WhisperX STT Server

WhisperX 기반 음성 처리 API. STT + 화자분리 + 발화분리 + PII 마스킹.

## 기술 스택

- Python 3.12 / FastAPI / Uvicorn
- WhisperX 3.8.5 (large-v3, CUDA)
- pyannote (화자분리, HF_TOKEN 필요)
- numpy + soundfile (오디오 처리)
- pytest (테스트)

## 필수 명령어

```bash
./run.sh dev              # dev 서버 (port 8001)
./run.sh live             # live 서버 (port 8000)
python -m pytest -q       # 테스트 실행 (TESTING=1 자동 적용)
sudo systemctl restart voice-api@dev   # 서비스 재시작
sudo systemctl restart voice-api@live
```

## 디렉토리 구조

```
app/
├── main.py              # FastAPI 진입점
├── config.py            # 환경변수 설정
├── stt_processor.py     # STT 파이프라인 (핵심)
├── pii_masker.py        # PII 마스킹
├── routers/             # health, transcribe
├── services/            # audio_preprocessor, audio_splitter, pii_service, utterance_segmenter, whisperx_service
├── models/schemas.py    # Pydantic 스키마
└── core/                # job_store (TTL 1h, max 100), exceptions
tests/                   # pytest
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | /api/v1/health | 서버 상태 |
| POST | /api/v1/transcribe | 음성 업로드 → STT |
| GET | /api/v1/jobs/{task_id} | 결과 조회 |
| GET | /api/v1/jobs/{task_id}/audio/{filename} | WAV 다운로드 |

## 청크 모드 응답 규약

원본 길이 ≥ `CHUNK_THRESHOLD_SEC`(기본 1h) 파일은 청크 모드로 처리되며, 응답에서 다음과 같이 동작한다:

- `chunked_processing: true`
- `utterances`: **청크 모드에서도 제공**. 각 청크가 메모리에 있는 동안 chunk-local 좌표로 발화 경계를 산출한 뒤 `extract_utterance_audio_local`로 WAV를 즉시 생성하고, 메타데이터 타임스탬프만 누적 offset으로 globalize한다. 청크 경계에 걸치는 발화는 경계에서 강제 분리될 수 있다.
- `speaker_audio`: **청크 모드에서는 `null` 고정**. 화자별 WAV는 전체 오디오 배열이 필요해 OOM 제약 때문에 제공하지 않는다.
- `_audio_files`: `utterance_XXX.wav`만 포함 (speaker_XX.wav 없음)

## 상세 참조

- `docs/api-reference.md` — API 상세 + Swagger 필드 설명
- `.claude/rules/python/stt-pipeline.md` — STT 파이프라인 아키텍처 + 비동기 작업 패턴
- `.claude/rules/python/performance.md` — 성능 최적화 + DeepFilterNet + 청크 모드
- `.claude/rules/python/config-and-pii.md` — 환경변수 테이블 + PII 마스킹 상세
- Swagger UI: `http://{host}:{port}/docs`
