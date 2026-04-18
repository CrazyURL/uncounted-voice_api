# Handoff Document
생성일시: 2026-04-19 KST
effort: high

## 1. 완료한 작업

- **로컬 게인 정규화 추가** (`audio_preprocessor.py`): 500ms 슬라이딩 윈도우 기반 `local_normalize_gain()` 신규 구현. 글로벌 10x 제한 이후에도 조용한 구간을 최대 30x로 부스트해 VAD 감지 개선.
- **pyannote VAD 임계값 주입 우회** (`stt_processor.py`): whisperx Pyannote.__init__가 vad_onset/vad_offset을 load_vad_model()에 전달하지 않는 버그를 `__new__` + 직접 vad_pipeline 주입으로 우회. VAD_ONSET=0.150, VAD_OFFSET=0.100 적용.
- **hanging word 보정** (`utterance_segmenter.py`): 발화 끝 단어가 0.3초 이상 gap 이후 고립된 경우 다음 발화 앞으로 이동하는 `_fix_hanging_words()` 추가.
- **cascade merge 버그 수정** (`utterance_segmenter.py`): 같은 화자 발화가 무한 병합되던 버그 수정. `last.duration < MIN_UTTERANCE_SEC` 조건 추가.
- **신규 config 변수 5개** (`config.py`): DIARIZATION_MODEL, HANGING_WORD_GAP_SEC, LOCAL_MAX_GAIN_X, VAD_ONSET, VAD_OFFSET
- **테스트 추가** (`test_audio_preprocessor.py`, `test_utterance_segmenter.py`): 16 + 5 = 21개 신규 테스트

## 2. 변경 파일 요약

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `app/config.py` | 수정 | 5개 env var 추가 |
| `app/services/audio_preprocessor.py` | 수정 | local_normalize_gain() + preprocess 파이프라인 연결 |
| `app/services/utterance_segmenter.py` | 수정 | _fix_hanging_words() + cascade merge 버그 수정 |
| `app/stt_processor.py` | 수정 | VAD 임계값 주입 우회 + DIARIZATION_MODEL config 적용 |
| `tests/test_audio_preprocessor.py` | 수정 | TestNormalizeGain, TestLocalNormalizeGain, TestGainPipelineIntegration (16 tests) |
| `tests/test_utterance_segmenter.py` | 수정 | TestCascadeMergeGuard, TestFixHangingWords (5 tests) |

## 3. 테스트 필요 사항

- [ ] pytest 전체 통과 확인
- [ ] local_normalize_gain: 조용한 구간 부스트 + 무음 과증폭 방지
- [ ] _fix_hanging_words: 이동 조건 정확성
- [ ] cascade merge guard: 무한 병합 방지

## 4. 알려진 이슈 / TODO

- [ ] VAD 주입 우회가 whisperx 업데이트 시 깨질 수 있음 (try/except 폴백 미구현)
- [ ] local_normalize_gain의 logger.info가 INFO 레벨로 과다 출력됨 (→ DEBUG 권장)
- [ ] config-and-pii.md 문서에 신규 env var 5개 미반영

## 5. 주의사항

- VAD 주입은 whisperx 내부 API 의존 — 업데이트 시 확인 필요
- _fix_hanging_words는 루프 중 result[i+1]을 수정 후 다음 반복에서 재사용 (캐스케이드 의도적)
- GPU 서버에서만 실제 STT 동작 확인 가능 — pytest는 로컬/CI에서만

## 6. 검증 권장 설정

- effort: high
- security: false
- coverage: true
- only: all
- loop: 3
