# 음성 PII 자동 마스킹 — 실행계획

> **Status:** 결정 완료 (2026-04-22) — Phase 1 착수 가능
> **Date:** 2026-04-22
> **Scope:** uncounted-voice-api

---

## 1. 현재 상태 요약

### 1.1 구현된 것 — 텍스트 PII 마스킹
- `app/pii_masker.py`에 regex 기반 한국어 PII 마스킹 구현
- 대상: 주민번호, 운전면허, 여권, 카드번호, 이메일, 전화번호, 계좌번호, IP, (옵션) 한국 이름
- `mask_segments(segments)`가 `stt_processor.py:644`에서 `seg["text"]`를 in-place로 교체
- 결과는 `pii_summary` (타입+카운트) 형태로 응답에 포함

### 1.2 **누락** — 음성(오디오) PII 마스킹
- 원본 오디오 자체에 PII 음성 구간이 **그대로 남아있음**
- `speaker_audio`, `utterance_audio` WAV 모두 PII 구간이 들려주는 상태
- 즉, **텍스트는 ***로 마스킹되지만 음성은 원본 그대로** 재생 가능

### 1.3 용어 정의
- **텍스트 PII 마스킹**: transcript 문자열의 PII 치환 (현재 구현됨)
- **음성 PII 마스킹**: 원본 PCM에서 PII가 발화된 시간 구간을 묵음/비프로 치환 (**미구현**)

---

## 2. 목표 기능

PII가 검출된 단어의 시간 범위(start/end)를 WhisperX word alignment에서 가져와,
해당 구간을 원본 오디오에서 묵음·비프·노이즈로 치환한 결과 WAV를 반환한다.

### 2.1 요구사항
1. STT 이후 검출된 PII regex match의 character position → word timestamp mapping
2. Matched word 구간(+optional buffer)을 원본 `audio` ndarray에서 치환
3. 치환된 오디오는 기존 `speaker_audio`/`utterance_audio` 생성 로직에 그대로 반영
4. 새 파라미터: `mask_audio_pii: bool = False` (기본 off, opt-in)
5. 응답 필드: `pii_summary[].time_ranges: [{start, end}]` 추가 (감사 로그용)
6. 테스트: 가짜 PII가 포함된 오디오 → 마스킹 구간 정확도 (±PAD_SEC 허용)

### 2.2 제외 범위
- STT가 놓친 PII는 마스킹되지 않음 (known limitation, 문서화)
- Chunked mode(>1h)에서는 full `audio` ndarray가 없으므로 **청크-local** 시점에서 마스킹 적용
- 한국어 이름의 audio 마스킹은 false-positive 리스크가 크므로 별도 flag(`mask_audio_names`) 로 분리

---

## 3. 기술 설계

### 3.1 데이터 흐름

```
WhisperX 추론 → align (word-level start/end)
             ↓
     _clean_segments() → segments[].words[]
             ↓
     [새 로직] PII detect with spans (text → char span)
             ↓
     [새 로직] char span → word index mapping
             ↓
     [새 로직] word timestamps → audio sample ranges
             ↓
     [새 로직] audio sample masking (silence/beep)
             ↓
   기존 audio_splitter (speaker/utterance) — masked audio 기반
             ↓
     mask_segments() (기존, 텍스트)
             ↓
     Response: audio_files[] + pii_summary (time_ranges 포함)
```

### 3.2 핵심 컴포넌트

#### 3.2.1 `pii_masker.py` 확장
현재 `mask_pii()`는 카운트만 반환. 다음 함수 추가:

```python
def detect_pii_spans(text: str, enable_name_masking: bool = False) -> list[dict]:
    """Return list of {type, char_start, char_end, matched_text}."""
```

기존 `mask_pii()`와 공존. `mask_pii()`는 내부적으로 `detect_pii_spans()` 호출 후 치환.

#### 3.2.2 새 모듈 `audio_pii_masker.py`
```python
def find_pii_word_ranges(
    segments: list[dict],
    enable_name_masking: bool = False,
    pad_sec: float = 0.15,
) -> list[tuple[float, float, str]]:
    """각 segment.text의 PII span을 segment.words의 start/end로 매핑한다.

    Returns: [(start_sec, end_sec, pii_type), ...]
    - char span이 걸치는 모든 word를 포함
    - pad_sec만큼 양쪽 버퍼 추가 (발음 꼬리 보호)
    """

def mask_audio_ranges(
    audio: np.ndarray,
    ranges: list[tuple[float, float, str]],
    sr: int,
    method: str = "silence",  # "silence" | "beep" | "pink_noise"
) -> np.ndarray:
    """지정 구간을 묵음/비프/노이즈로 치환한다. 원본 audio는 불변 (copy)."""
```

#### 3.2.3 `stt_processor.py` 통합
- Line 644 근처에 `mask_audio_pii` 플래그 분기 추가
- 일반 모드: `audio = mask_audio_ranges(audio, ranges, ...)` 이후 speaker/utterance split 수행
- 청크 모드: `_transcribe_chunked` 내부에서 각 `chunk_audio`에 동일 로직 적용

### 3.3 Char-span → Word timestamp 매핑 알고리즘

WhisperX word는 `text`의 문자 offset을 직접 제공하지 않는다. 따라서:

1. `segment.text`를 `segment.words[].word`를 순서대로 조인하면서 누적 char offset 계산
2. (공백 허용) `text.find(word, last_end)`로 각 word의 char span 찾기
3. PII match의 `[char_start, char_end]`가 걸치는 word 인덱스 집합 산출
4. 해당 words의 최소 start / 최대 end를 시간 범위로 사용

대안(더 간단): PII 정규식을 `segment.text` 대신 `segment.words[].word` 연결 결과에 매칭.
→ 공백 처리 차이 존재, 별도 검증 필요.

### 3.4 마스킹 방법 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| Silence (0-fill) | 단순, 파일 크기 최소 | "뚝 끊기는" 어색함, 공격자는 편집 감지 용이 |
| Beep (1kHz sine) | 명확한 마스킹 신호, 표준적 | 라우드니스 주의, fade 필요 |
| Pink noise | 자연스러움, 편집 감지 어려움 | 여전히 PII 추출 시도 가능 우려 (음성 흔적 없음) |

**확정 기본값: 1kHz 비프 + 10ms fade in/out** — G-STAR 기획서(`generate_gstar_docx.py:247`)와 특허 출원 청구항 ①에 명시된 방법과 정합. Sine 1kHz 톤 생성 후 양 끝 10ms 선형 fade로 클릭 노이즈 방지.

---

## 4. 구현 단계 (Phase 분할)

### Phase 1 — pii_masker 확장 (위치 정보 반환)
- [ ] `detect_pii_spans()` 신규 함수
- [ ] 기존 `mask_pii()`가 내부에서 `detect_pii_spans()`를 호출하도록 리팩터 (동작 보존)
- [ ] Unit test: 알려진 PII 문자열 → span 정확도

**산출물:** `pii_masker.py` 수정 + `tests/test_pii_spans.py`
**예상 LOC:** +80 lines, 수정 20 lines
**의존성:** 없음

### Phase 2 — audio_pii_masker 모듈
- [ ] `find_pii_word_ranges()` — segment spans → word time ranges
- [ ] `mask_audio_ranges()` — ndarray 치환 (silence + fade)
- [ ] Unit test: 합성 오디오(sine + 자른 word) → 정확히 마스킹되는지

**산출물:** `app/services/audio_pii_masker.py` + `tests/test_audio_pii_masker.py`
**예상 LOC:** +150 lines
**의존성:** Phase 1

### Phase 3 — stt_processor 통합 (일반 모드)
- [ ] `transcribe()` 시그니처에 `mask_audio_pii: bool = False` 추가
- [ ] Line 644 직후 일반 모드에 audio masking 적용
- [ ] 응답에 `pii_summary[].time_ranges` 포함
- [ ] Integration test: 실제 샘플 오디오 → API 호출 → 마스킹 확인

**산출물:** `stt_processor.py`, `app/models/schemas.py`, `routers/transcribe.py` 수정
**예상 LOC:** +60 lines
**의존성:** Phase 2

### Phase 4 — 청크 모드 지원
- [ ] `_transcribe_chunked()` 루프 내에서 각 `chunk_audio`에 적용
- [ ] `emit_chunk_utterances()` 이전에 masking 수행 (utterance WAV도 마스킹 반영)
- [ ] Integration test: >1h 합성 오디오 → 각 청크에 PII 삽입 후 검증

**산출물:** `stt_processor.py`, `chunk_utterance_emitter.py` 호출 순서 조정
**예상 LOC:** +30 lines
**의존성:** Phase 3

### Phase 5 — 문서화 & 운영
- [ ] `docs/api-reference.md`에 `mask_audio_pii` 파라미터 추가
- [ ] `CLAUDE.md` / `config-and-pii.md`에 기능 설명 추가
- [ ] `PAD_SEC`, `PII_MASK_METHOD` 환경변수 노출 (튜닝 가능)
- [ ] 한계 명시: "STT가 놓친 PII는 마스킹되지 않음"

**산출물:** 문서 3개 수정
**예상 LOC:** +50 lines (docs)
**의존성:** Phase 4

---

## 5. 테스트 계획

### 5.1 Unit
- `detect_pii_spans`: 각 패턴별 char span 정확도
- `find_pii_word_ranges`: 가짜 segments → 기대 word ranges
- `mask_audio_ranges`: 합성 sine wave의 특정 구간만 0이 되는지 sample-level 검증

### 5.2 Integration
- 실제 WhisperX 파이프라인 전체: 전화번호 포함 오디오 → API 응답의 WAV를 다시 로드 → 해당 구간 RMS ≈ 0
- Chunked mode: 1.5h 합성 오디오 + 경계 근처 PII → 모든 구간 마스킹 확인

### 5.3 Regression
- 기존 178개 테스트 그대로 통과
- `mask_audio_pii=False` 일 때 기존 응답 byte-equivalent

---

## 6. 리스크 & 제약

| 리스크 | 영향 | 대응 |
|--------|------|------|
| STT가 PII를 잘못 인식하여 missing | 음성 PII 잔존 (정보 유출) | 문서에 한계 명시, 수동 검토 권장 |
| Word timestamp 부정확 → 마스킹 위치 오차 | PII 일부 들림 | `PAD_SEC` 기본 0.15s 버퍼 + 튜닝 가능 |
| 청크 경계에 PII 걸친 경우 | 한쪽 청크에만 마스킹 | chunk overlap 현재 0 — 드문 케이스, 문서 명시 |
| 청크 모드에서 full audio 없음 | 구조 변경 필요 | Phase 3(일반) → Phase 4(청크) 분리 |
| 한국어 이름 false positive | 일반 단어 마스킹 | `mask_audio_names`를 기본 off, 별도 flag |
| VRAM/메모리 증가 | audio.copy()로 배수 증가 | in-place 마스킹 옵션 또는 mmap 검토 (후순위) |

---

## 7. 확정된 결정 (2026-04-22)

### Q1. 비즈니스 규칙 충돌 여부 — **A. 인프라 배치 문제로 간주**
- CLAUDE.md #1 "GPU 기반 AI 추론 결과 저장/표시/판매 금지"는 감정/스트레스 등 **프로파일링 판정값**이 대상
- 음성 PII 마스킹은 오히려 비즈니스 모델의 **핵심 자산** (G-STAR 기획서·특허 청구항 ①)
- 기획서 원안 "온디바이스"와 현재 서버(voice-api) 구현 차이는 **자체 인프라 배치**로 해석 (제3자 클라우드 아님)
- 사용자는 `mask_audio_pii`를 opt-in하여 동의 흐름을 유지

### Q2. 마스킹 방법 — **A. 1kHz 비프 (+10ms fade)**
- 이유: G-STAR 기획서 `:247` 및 특허 청구항과 정합
- 환경변수 미노출 (고정). 추후 변경 필요 시 별도 PR로 확장

### Q3. 결과 파일 정책 — **A. 플래그 기반 덮어쓰기**
- `mask_audio_pii=True`: `speaker_audio`/`utterance_audio` WAV를 처음부터 마스킹된 상태로 생성
- `mask_audio_pii=False` (기본): 현재 동작 보존 (byte-equivalent)
- 별도 `*_masked.wav` 저장 없음 (저장 절반, 정책 단순)

### Q4. 한국 이름 오디오 마스킹 — **A. 별도 `mask_audio_names` 플래그**
- 텍스트의 `enable_name_masking`와 분리
- 기본값 `False` (audio 쪽 false positive 영향이 더 크므로 보수적)
- `mask_audio_pii=True`일 때만 의미 있음

### Q5. 버퍼 PAD_SEC 기본값 — **B. 0.15s**
- 계획 문서 초기 권장값 확정
- 환경변수 `PII_MASK_PAD_SEC`로 튜닝 가능 (운영 중 조정)

### Q6. 배포 절차 (기본 채택)
- dev 서버(`voice-api@dev`, port 8001)에서 검증 → live(`voice-api@live`, port 8000) 승격
- 롤백: 플래그 off가 기본이므로 기존 동작 유지

---

## 확정된 기술 스펙 요약

| 항목 | 값 |
|------|-----|
| 마스킹 방법 | 1kHz sine + 10ms fade |
| 기본 플래그 | `mask_audio_pii=False`, `mask_audio_names=False` |
| 버퍼 | `PAD_SEC = 0.15s` (env tunable) |
| 결과 정책 | 플래그 ON 시 `speaker_audio`/`utterance_audio` 덮어쓰기 |
| 응답 필드 추가 | `pii_summary[].time_ranges: [{start, end}]` |
| 배포 순서 | dev → live |

---

## 8. 다음 단계

1. ✅ Q1~Q6 결정 확정 (2026-04-22)
2. `prompt_plan.md`에 Phase 1~5 항목 추가
3. Phase 1부터 TDD 순차 진행 (RED → GREEN → REFACTOR)
4. Phase 3 완료 시 dev 서버(`voice-api@dev`) end-to-end 검증
5. Phase 5 완료 후 live 서버(`voice-api@live`) 배포

---

## 부록 A — 영향 파일 목록 (예상)

| 파일 | 변경 유형 |
|------|----------|
| `app/pii_masker.py` | 수정 (span 반환 추가) |
| `app/services/audio_pii_masker.py` | 신규 |
| `app/stt_processor.py` | 수정 (통합) |
| `app/models/schemas.py` | 수정 (응답 필드) |
| `app/routers/transcribe.py` | 수정 (param) |
| `tests/test_pii_spans.py` | 신규 |
| `tests/test_audio_pii_masker.py` | 신규 |
| `tests/test_stt_processor_audio_pii.py` | 신규 (integration) |
| `docs/api-reference.md` | 수정 |
| `.claude/rules/python/config-and-pii.md` | 수정 |
| `CLAUDE.md` | 수정 (기능 설명 1줄) |

## 부록 B — 추정 작업량

| Phase | 작업량 |
|-------|-------|
| Phase 1 | 반일 |
| Phase 2 | 1일 |
| Phase 3 | 반일 |
| Phase 4 | 반일 |
| Phase 5 | 반일 |
| **총계** | **약 3일** (1인 기준, 테스트 포함) |
