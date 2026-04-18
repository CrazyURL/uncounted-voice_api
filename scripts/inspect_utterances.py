"""발화 분리 결과 확인 스크립트.

사용법:
    HF_TOKEN=<token> python scripts/inspect_utterances.py <wav_file>

예시:
    HF_TOKEN=hf_xxx python scripts/inspect_utterances.py sample_data/utt_34c810c30650d12e_008.wav
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def main():
    if len(sys.argv) < 2:
        print("사용법: python scripts/inspect_utterances.py <wav_file>", file=sys.stderr)
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"[ERROR] 파일 없음: {src}", file=sys.stderr)
        sys.exit(1)

    # 원본 보존을 위해 임시 복사본으로 처리 (transcribe가 원본 삭제)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(src, tmp_path)

    from app.stt_processor import load_models, transcribe

    print("모델 로딩 중...", flush=True)
    load_models()

    print(f"\n처리 중: {src.name}", flush=True)
    result = transcribe(
        file_path=str(tmp_path),
        task_id="inspect_utterances",
        enable_diarize=True,
        enable_name_masking=False,
        mask_pii=False,
        split_by_speaker=False,
        split_by_utterance=True,
        denoise_enabled=False,
    )

    utterances = result.get("utterances", [])
    duration = result.get("duration_seconds", 0)

    print(f"\n{'─' * 60}")
    print(f"파일: {src.name}  ({duration:.2f}s)")
    print(f"발화 수: {len(utterances)}개")
    print(f"{'─' * 60}")

    for i, u in enumerate(utterances, 1):
        print(
            f"[{i:02d}] {u['speaker_id']:<12} "
            f"{u['start_sec']:6.2f}s ~ {u['end_sec']:6.2f}s  "
            f"({u['duration_sec']:.2f}s)"
        )
        print(f"       {u['transcript_text'][:80]}")

    print(f"{'─' * 60}")

    # 발화별 WAV 저장
    audio_files: dict = result.get("_audio_files", {})
    if audio_files:
        out_dir = src.parent / src.stem
        out_dir.mkdir(exist_ok=True)
        for filename, wav_bytes in audio_files.items():
            if isinstance(wav_bytes, bytes):
                (out_dir / filename).write_bytes(wav_bytes)
        print(f"\n발화 WAV 저장: {out_dir}/ ({len(audio_files)}개)")
    else:
        print("\n발화 WAV 없음 (audio_files 비어 있음)")

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, bytes):
                return f"<bytes len={len(o)}>"
            return super().default(o)

    out_path = src.with_suffix(".result.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, cls=_Encoder))
    print(f"JSON 저장: {out_path}")


if __name__ == "__main__":
    main()
