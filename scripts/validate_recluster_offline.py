#!/usr/bin/env python3
"""Phase 4-6 실오디오 오프라인 검증.

sample_data/ 디렉토리에 저장된 이전 voice-api 처리 결과(task JSON + 발화별
utterance_XXX.wav)를 이용해 Phase 4-6 (pyannote 임베딩 → 윈도우 빌더 →
AHC 재클러스터링)을 end-to-end로 검증한다.

원본 m4a가 없어도 된다. utterance wav를 word 타임스탬프에 맞춰 붙여 넣은
pseudo audio 배열을 구성하면, Phase 5의 윈도우 슬라이스는 실제 음성 영역에
떨어진다 (발화 사이 silence는 0으로 패딩됨).

흐름:
    1. sample_data/*.json 순회
    2. 각 task에 대해:
       a. utterances → flat words + pseudo audio 재구성
       b. maybe_recluster_speakers(audio, 16k, words, ...) 호출
       c. 라벨 분포 비교 (baseline vs reclustered)
    3. JSON 레코드 + markdown 리포트 저장

사용법:
    cd uncounted-voice-api
    source venv/bin/activate
    python scripts/validate_recluster_offline.py
    python scripts/validate_recluster_offline.py --tasks 6c821aaf5d50,5dc13ea73c62
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.services import speaker_recluster  # noqa: E402
from app.services.speaker_embedding import SpeakerEmbeddingModel  # noqa: E402
from app.services.speaker_recluster import maybe_recluster_speakers  # noqa: E402

SAMPLE_RATE = 16_000
SAMPLE_DIR = REPO_ROOT / "sample_data"
_OUT_BASE = REPO_ROOT / ".validation"


def _load_task_json(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text())


def _flatten_words(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for utt in utterances:
        utt_speaker = utt.get("speaker_id") or utt.get("speaker") or "SPEAKER_00"
        for w in utt.get("words") or []:
            flat.append(
                {
                    "word": w.get("word") or w.get("text") or "",
                    "start": float(w.get("start") or 0.0),
                    "end": float(w.get("end") or 0.0),
                    "speaker_id": w.get("speaker") or w.get("speaker_id") or utt_speaker,
                    "speaker": w.get("speaker") or utt_speaker,
                }
            )
    return flat


def _load_real_audio(audio_path: Path) -> np.ndarray:
    """ffmpeg로 m4a/mp3/wav를 16k mono float32로 디코드."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(audio_path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _reconstruct_pseudo_audio(task_dir: Path, utterances: list[dict[str, Any]]) -> np.ndarray:
    """각 utterance_XXX.wav를 utterance.start_sec 위치에 paste한 mono float32 array."""
    max_end = 0.0
    for u in utterances:
        end = float(u.get("end_sec") or 0.0)
        for w in u.get("words") or []:
            end = max(end, float(w.get("end") or 0.0))
        max_end = max(max_end, end)

    total_samples = int((max_end + 1.0) * SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float32)

    for u in utterances:
        filename = u.get("audio_filename")
        if not filename:
            continue
        wav_path = task_dir / filename
        if not wav_path.exists():
            continue
        try:
            data, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed to load {wav_path.name}: {e}", file=sys.stderr)
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr
            new_len = int(len(data) * ratio)
            old_idx = np.arange(len(data))
            new_idx = np.linspace(0, len(data) - 1, new_len)
            data = np.interp(new_idx, old_idx, data).astype(np.float32)
        start_sec = float(u.get("start_sec") or 0.0)
        start_sample = int(start_sec * SAMPLE_RATE)
        end_sample = start_sample + len(data)
        if end_sample > len(audio):
            audio = np.pad(audio, (0, end_sample - len(audio) + 1))
        audio[start_sample:end_sample] = data
    return audio


def _label_distribution(words: list[dict[str, Any]], key: str = "speaker_id") -> dict[str, int]:
    dist: dict[str, int] = {}
    for w in words:
        k = w.get(key) or w.get("speaker") or "UNKNOWN"
        dist[k] = dist.get(k, 0) + 1
    return dist


def _permutation_invariant_match(
    baseline: list[dict[str, Any]], post: list[dict[str, Any]]
) -> tuple[float, dict[str, str]]:
    """2-화자 최적 permutation으로 identity 매칭률 계산.

    baseline과 post는 길이/순서가 동일하다고 가정 (1:1 word 매칭).
    반환: (best_accuracy, best_mapping) — mapping은 post label → baseline label.
    """
    if len(baseline) != len(post) or not baseline:
        return 0.0, {}

    def _lab(w: dict[str, Any]) -> str:
        return w.get("speaker_id") or w.get("speaker") or "UNKNOWN"

    base_labels = [_lab(b) for b in baseline]
    post_labels = [_lab(p) for p in post]
    unique_post = sorted(set(post_labels))
    unique_base = sorted(set(base_labels))

    total = len(base_labels)
    best_acc = 0.0
    best_map: dict[str, str] = {}

    targets = unique_base if len(unique_base) >= len(unique_post) else unique_base + unique_post
    targets = list(dict.fromkeys(targets))

    for perm in itertools.permutations(targets, r=len(unique_post)):
        mapping = dict(zip(unique_post, perm))
        matches = sum(1 for p, b in zip(post_labels, base_labels) if mapping.get(p, p) == b)
        acc = matches / total
        if acc > best_acc:
            best_acc = acc
            best_map = mapping

    return best_acc, best_map


def _validate_task(
    task_id: str,
    json_path: Path,
    model: SpeakerEmbeddingModel,
    real_audio_path: Path | None = None,
) -> dict[str, Any]:
    print(f"\n[{task_id}] loading {json_path.name}")
    data = _load_task_json(json_path)
    utterances = data.get("utterances") or []
    if not utterances:
        return {"task_id": task_id, "error": "no utterances"}

    task_dir = json_path.parent / task_id
    words = _flatten_words(utterances)
    if not words:
        return {"task_id": task_id, "error": "no words"}

    print(f"[{task_id}] {len(utterances)} utterances, {len(words)} words")
    baseline_dist = _label_distribution(words)
    print(f"[{task_id}] baseline dist: {baseline_dist}")

    audio_mode: str
    t0 = time.time()
    if real_audio_path is not None:
        print(f"[{task_id}] loading REAL audio: {real_audio_path.name}")
        audio = _load_real_audio(real_audio_path)
        audio_mode = "real"
    else:
        print(f"[{task_id}] reconstructing pseudo-audio...")
        audio = _reconstruct_pseudo_audio(task_dir, utterances)
        audio_mode = "pseudo"
    print(
        f"[{task_id}] audio[{audio_mode}] samples={len(audio)} "
        f"({len(audio)/SAMPLE_RATE:.1f}s) load={time.time()-t0:.1f}s"
    )

    print(f"[{task_id}] calling maybe_recluster_speakers (pyannote lazy load)...")
    t1 = time.time()
    try:
        result = maybe_recluster_speakers(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            words=words,
            segments=[],
            mode="call_recording",
            embedding_model=model,
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return {"task_id": task_id, "error": f"recluster exception: {e}"}
    elapsed = time.time() - t1

    post_words = list(result.words)
    new_dist = _label_distribution(post_words)
    print(
        f"[{task_id}] recluster: window_count={result.window_count} "
        f"confidence={result.confidence:.3f} changed={result.changed} "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"[{task_id}] post dist: {new_dist}")

    # Raw changed count (label-sensitive)
    raw_changed = sum(
        1
        for before, after in zip(words, post_words)
        if before["speaker_id"] != after.get("speaker_id")
    )
    # Permutation-invariant match accuracy
    best_acc, best_map = _permutation_invariant_match(words, post_words)
    semantic_changed = int(round((1.0 - best_acc) * len(words)))
    print(
        f"[{task_id}] perm-invariant acc={best_acc:.3f} "
        f"(semantic_changed={semantic_changed}, raw_changed={raw_changed}) "
        f"map={best_map}"
    )

    return {
        "task_id": task_id,
        "audio_mode": audio_mode,
        "utterance_count": len(utterances),
        "word_count": len(words),
        "audio_duration_sec": len(audio) / SAMPLE_RATE,
        "baseline_distribution": baseline_dist,
        "post_recluster_distribution": new_dist,
        "window_count": result.window_count,
        "confidence": float(result.confidence),
        "changed": bool(result.changed),
        "raw_changed_words": raw_changed,
        "perm_invariant_accuracy": round(best_acc, 4),
        "semantic_changed_words": semantic_changed,
        "best_label_mapping": best_map,
        "elapsed_sec": round(elapsed, 2),
    }


def _discover_tasks(selected: list[str] | None) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for p in sorted(SAMPLE_DIR.glob("*.json")):
        task_id = p.stem
        if selected and task_id not in selected:
            continue
        task_dir = SAMPLE_DIR / task_id
        if task_dir.exists() and task_dir.is_dir():
            pairs.append((task_id, p))
    return pairs


def _render_report(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Phase 4-6 Offline Validation Report")
    lines.append(f"generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    lines.append(f"sample_dir: {SAMPLE_DIR}")
    lines.append("")
    lines.append(
        "| task | words | audio(s) | windows | confidence | changed | raw Δ | perm-acc | semantic Δ | elapsed(s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        if "error" in r:
            lines.append(f"| {r['task_id']} | ERROR: {r['error']} |||||||||")
            continue
        lines.append(
            f"| {r['task_id']} | {r['word_count']} | "
            f"{r['audio_duration_sec']:.1f} | {r['window_count']} | {r['confidence']:.3f} | "
            f"{r['changed']} | {r['raw_changed_words']} | "
            f"{r['perm_invariant_accuracy']:.3f} | {r['semantic_changed_words']} | "
            f"{r['elapsed_sec']} |"
        )
    lines.append("")
    lines.append("## Per-task distribution")
    for r in records:
        if "error" in r:
            continue
        lines.append(f"\n### {r['task_id']}")
        lines.append("```")
        lines.append(f"baseline:    {json.dumps(r['baseline_distribution'], ensure_ascii=False)}")
        lines.append(f"post:        {json.dumps(r['post_recluster_distribution'], ensure_ascii=False)}")
        lines.append("```")
    return "\n".join(lines)


OUT_DIR: Path = _OUT_BASE / "recluster_offline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="comma-separated task ids (default: all in sample_data)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override Phase 6 confidence_threshold (monkey-patches _DEFAULT_CONFIDENCE_THRESHOLD)",
    )
    parser.add_argument(
        "--real-audio",
        type=str,
        default="",
        help="TASK_ID:PATH to load real audio via ffmpeg (overrides pseudo-audio for that task)",
    )
    args = parser.parse_args()
    selected = [t.strip() for t in args.tasks.split(",") if t.strip()] or None

    if args.threshold is not None:
        print(f"[threshold] override {speaker_recluster._DEFAULT_CONFIDENCE_THRESHOLD} → {args.threshold}")
        speaker_recluster._DEFAULT_CONFIDENCE_THRESHOLD = args.threshold

    threshold = speaker_recluster._DEFAULT_CONFIDENCE_THRESHOLD
    global OUT_DIR
    OUT_DIR = _OUT_BASE / f"recluster_offline_th{threshold:.2f}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[out] writing to {OUT_DIR}")

    pairs = _discover_tasks(selected)
    if not pairs:
        print("no task JSON + dir pairs found in sample_data/", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(pairs)} task(s): {[t[0] for t in pairs]}")

    real_audio_map: dict[str, Path] = {}
    if args.real_audio:
        try:
            tid, p = args.real_audio.split(":", 1)
            real_audio_map[tid.strip()] = Path(p.strip())
        except ValueError:
            print(f"--real-audio expects TASK_ID:PATH (got {args.real_audio})", file=sys.stderr)
            sys.exit(2)
        if not real_audio_map[tid].exists():
            print(f"real audio not found: {real_audio_map[tid]}", file=sys.stderr)
            sys.exit(2)

    model = SpeakerEmbeddingModel()  # lazy loaded on first call

    records: list[dict[str, Any]] = []
    for task_id, json_path in pairs:
        try:
            records.append(
                _validate_task(task_id, json_path, model, real_audio_path=real_audio_map.get(task_id))
            )
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            records.append({"task_id": task_id, "error": str(e)})

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nwrote {summary_path}")

    report_path = OUT_DIR / "report.md"
    report_path.write_text(_render_report(records))
    print(f"wrote {report_path}")
    print("\n" + "=" * 60)
    print(_render_report(records))


if __name__ == "__main__":
    main()
