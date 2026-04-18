"""assign_word_speakers 직후 단어별 speaker 매핑 확인 스크립트.

pyannote가 2 화자를 올바르게 검출했는데도 assign_word_speakers 후
전부 SPEAKER_00 이 되는 원인을 추적한다.

사용법:
    source .env.dev && python scripts/debug_assign_speakers.py <wav_file>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import whisperx
import app.stt_processor as proc
from app import config


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/debug_assign_speakers.py <wav_file>", file=sys.stderr)
        sys.exit(1)

    wav_path = Path(sys.argv[1]).resolve()
    if not wav_path.exists():
        print(f"[ERROR] 파일 없음: {wav_path}", file=sys.stderr)
        sys.exit(1)

    print("모델 로딩 중...", flush=True)
    proc.load_models()

    if proc._diarize_model is None:
        print("[ERROR] 화자분리 모델 로드 실패 (HF_TOKEN 확인)", file=sys.stderr)
        sys.exit(1)

    print(f"오디오 로드: {wav_path.name}", flush=True)
    audio = whisperx.load_audio(str(wav_path))
    duration = len(audio) / 16000
    print(f"길이: {duration:.2f}s\n")

    # ── 1. Transcribe ──
    print("── 1. Transcribe ──")
    result = proc._model.transcribe(audio, batch_size=config.BATCH_SIZE)
    print(f"세그먼트 수: {len(result['segments'])}")
    for i, seg in enumerate(result["segments"]):
        words = seg.get("words", [])
        print(f"  [{i}] {seg['start']:.2f}~{seg['end']:.2f}  '{seg['text'].strip()}'  ({len(words)} words)")
    print()

    # ── 2. Align ──
    print("── 2. Align ──")
    try:
        result = whisperx.align(
            result["segments"], proc._align_model, proc._align_metadata,
            audio, config.DEVICE, return_char_alignments=False,
        )
        print("Alignment 완료")
        print("단어별 타임스탬프 샘플 (처음 10개):")
        all_words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                all_words.append(w)
        for w in all_words[:10]:
            start = w.get("start", "?")
            end = w.get("end", "?")
            print(f"  {start}~{end}  '{w['word']}'")
    except Exception as e:
        print(f"[WARN] Alignment 실패: {e}")
    print()

    # ── 3. Diarize (raw) ──
    print("── 3. pyannote 화자분리 원시 출력 (min=2, max=2) ──")
    diarize_segments = proc._diarize_model(audio, min_speakers=2, max_speakers=2)
    print(diarize_segments)
    print()

    # ── 4. assign_word_speakers ──
    print("── 4. assign_word_speakers 결과 ──")
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)
    print("단어별 speaker:")
    for seg in result_with_speakers["segments"]:
        seg_speaker = seg.get("speaker", "NONE")
        for w in seg.get("words", []):
            start = w.get("start", "?")
            end = w.get("end", "?")
            word_speaker = w.get("speaker", "NONE")
            print(f"  {start:.2f}~{end:.2f}  [{word_speaker}]  '{w['word']}'")
        print(f"  → 세그먼트 speaker: {seg_speaker}  '{seg['text'].strip()}'")
        print()


if __name__ == "__main__":
    main()
