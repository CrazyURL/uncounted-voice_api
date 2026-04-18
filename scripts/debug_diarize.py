"""pyannote raw diarization 출력 확인 스크립트.

사용법:
    source .env.dev
    python scripts/debug_diarize.py <wav_file>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import soundfile as sf
import torch
from whisperx.diarize import DiarizationPipeline


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/debug_diarize.py <wav_file>", file=sys.stderr)
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    hf_token = os.environ.get("HF_TOKEN", "")
    diarization_model = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    force_two = os.environ.get("VOICE_DIARIZATION_FORCE_TWO_SPEAKERS", "false").lower() in ("true", "1", "yes")

    print(f"모델: {diarization_model}")
    print(f"force_two_speakers: {force_two}")

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"오디오: {len(audio)/sr:.2f}s, {sr}Hz\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DiarizationPipeline(model_name=diarization_model, token=hf_token, device=device)

    options = {"min_speakers": 2, "max_speakers": 2} if force_two else {}
    print(f"pyannote 옵션: {options}")

    diarize_segments = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr}, **options)

    print("\n── pyannote raw segments ──────────────────────────")
    print(diarize_segments.to_string())
    print("───────────────────────────────────────────────────\n")

    # 스피커별 타임라인
    speakers = diarize_segments["speaker"].unique()
    print(f"감지된 화자 수: {len(speakers)}명 ({', '.join(sorted(speakers))})\n")
    for spk in sorted(speakers):
        rows = diarize_segments[diarize_segments["speaker"] == spk]
        total = sum(r["end"] - r["start"] for _, r in rows.iterrows())
        print(f"  {spk}: {len(rows)}구간, 총 {total:.2f}s")
        for _, r in rows.iterrows():
            print(f"    {r['start']:.3f}s ~ {r['end']:.3f}s  ({r['end']-r['start']:.3f}s)")


if __name__ == "__main__":
    main()
