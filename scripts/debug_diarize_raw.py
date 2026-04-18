"""pyannote 화자분리 원시 출력 확인 스크립트.

사용법:
    source .env.dev && python scripts/debug_diarize_raw.py <wav_file>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import whisperx
import app.stt_processor as proc


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/debug_diarize_raw.py <wav_file>", file=sys.stderr)
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

    for label, opts in [
        ("힌트 없음 (기본)", {}),
        ("min=2, max=2", {"min_speakers": 2, "max_speakers": 2}),
        ("min=2 only",   {"min_speakers": 2}),
    ]:
        print(f"── {label} ──")
        try:
            segs = proc._diarize_model(audio, **opts)
            print(segs)
        except Exception as e:
            print(f"[ERROR] {e}")
        print()


if __name__ == "__main__":
    main()
