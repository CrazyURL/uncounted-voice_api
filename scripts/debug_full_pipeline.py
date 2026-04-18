"""실제 파이프라인 전체 시뮬레이션 — 발화 분리 결과 확인.

preprocess → transcribe → align → diarize → assign_word_speakers → segment_utterances
의 실제 경로를 그대로 재현해서, 각 단계 결과를 출력한다.

사용법:
    source .env.dev && python scripts/debug_full_pipeline.py <wav_file>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import whisperx
import app.stt_processor as proc
from app import config
from app.services.audio_preprocessor import preprocess
from app.services.utterance_segmenter import segment as segment_utterances


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/debug_full_pipeline.py <wav_file>", file=sys.stderr)
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

    # ── 0. 오디오 로드 + 전처리 ──
    print(f"\n오디오 로드: {wav_path.name}", flush=True)
    raw_audio = whisperx.load_audio(str(wav_path))
    print(f"raw 길이: {len(raw_audio)/16000:.2f}s")

    audio = preprocess(raw_audio, config.SAMPLE_RATE)
    print(f"전처리 후 길이: {len(audio)/16000:.2f}s\n")

    total_dur = len(audio) / config.SAMPLE_RATE

    # ── 1. Transcribe ──
    print("── 1. Transcribe ──")
    result = proc._model.transcribe(audio, batch_size=config.BATCH_SIZE)
    print(f"세그먼트 수: {len(result['segments'])}")
    for i, seg in enumerate(result["segments"]):
        print(f"  [{i}] {seg['start']:.2f}~{seg['end']:.2f}  '{seg['text'].strip()}'")
    print()

    # ── 2. Align ──
    print("── 2. Align ──")
    try:
        result = whisperx.align(
            result["segments"], proc._align_model, proc._align_metadata,
            audio, config.DEVICE, return_char_alignments=False,
        )
        print("완료")
    except Exception as e:
        print(f"[WARN] 실패: {e}")
    print()

    # ── 3. Diarize ──
    print("── 3. Diarize (min=2, max=2) ──")
    diarize_segments = proc._diarize_model(audio, min_speakers=2, max_speakers=2)
    print(diarize_segments)
    print()

    # ── 4. assign_word_speakers ──
    print("── 4. assign_word_speakers ──")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print("단어별 speaker:")
    for seg in result["segments"]:
        for w in seg.get("words", []):
            start = w.get("start", "?")
            end = w.get("end", "?")
            sp = w.get("speaker", "NONE")
            if isinstance(start, float):
                print(f"  {start:.2f}~{end:.2f}  [{sp}]  '{w['word']}'")
    print()

    # ── 5. _clean_segments ──
    print("── 5. _clean_segments 후 ──")
    segments = proc._clean_segments(result["segments"])

    # ── 6. all_words 구성 (stt_processor.py 동일 로직) ──
    all_words = []
    for s in segments:
        if s.get("words"):
            for w in s["words"]:
                all_words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", s["start"]),
                    "end": w.get("end", s["end"]),
                    "speaker": w.get("speaker", s.get("speaker")),
                })
        else:
            all_words.append({
                "word": s.get("text", ""),
                "start": s["start"],
                "end": s["end"],
                "speaker": s.get("speaker"),
            })

    # None speaker fallback
    for i, w in enumerate(all_words):
        if w["speaker"] is None:
            if i > 0 and all_words[i - 1]["speaker"] is not None:
                w["speaker"] = all_words[i - 1]["speaker"]
            elif i + 1 < len(all_words) and all_words[i + 1]["speaker"] is not None:
                w["speaker"] = all_words[i + 1]["speaker"]
            else:
                w["speaker"] = "SPEAKER_0"

    print("all_words (segment_utterances에 전달되는 단어):")
    for w in all_words:
        print(f"  {w['start']:.2f}~{w['end']:.2f}  [{w['speaker']}]  '{w['word']}'")
    print()

    # ── 7. segment_utterances ──
    print("── 7. segment_utterances 결과 ──")
    boundaries = segment_utterances(all_words, total_dur)
    for idx, utt in enumerate(boundaries):
        print(f"  [{idx:02d}] {utt.start_sec:.2f}~{utt.end_sec:.2f}  "
              f"speaker={utt.speaker_id}  '{utt.transcript_text}'")
    print()
    print(f"총 발화 수: {len(boundaries)}")


if __name__ == "__main__":
    main()
