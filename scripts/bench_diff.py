"""Baseline vs Turbo 결과 diff 계산.

WER(baseline, turbo)로 두 모델의 출력 차이를 측정한다.
샘플별 ground truth가 없으므로 baseline을 reference로 두고 turbo를 hypothesis로 계산.
2%p 미만 → 수용, 초과 → 수동 검토.

사용법:
    pip install jiwer  # 최초 1회
    python scripts/bench_diff.py \
        --baseline ~/voice-api-bench/results/baseline \
        --turbo ~/voice-api-bench/results/turbo \
        --out ~/voice-api-bench/results/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from jiwer import wer, cer
except ImportError:
    raise SystemExit("jiwer 미설치: pip install jiwer")


def load_transcript(result_path: Path) -> str:
    """result.json에서 세그먼트 텍스트를 하나로 이어붙인다."""
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", []) or data.get("utterances", [])
    parts = [(s.get("text") or "").strip() for s in segments]
    return " ".join(p for p in parts if p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--turbo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=2.0, help="WER diff 수용 임계 (퍼센트 포인트)")
    args = parser.parse_args()

    # 두 디렉토리 모두에 존재하는 hash만 비교
    baseline_samples = {p.stem.replace(".result", ""): p for p in args.baseline.glob("*.result.json")}
    turbo_samples = {p.stem.replace(".result", ""): p for p in args.turbo.glob("*.result.json")}
    common = sorted(set(baseline_samples.keys()) & set(turbo_samples.keys()))

    if not common:
        print("공통 샘플이 없습니다.")
        return 1

    print(f"공통 샘플: {len(common)}개\n")
    rows: list[dict[str, Any]] = []

    for hash_id in common:
        ref = load_transcript(baseline_samples[hash_id])
        hyp = load_transcript(turbo_samples[hash_id])

        if not ref or not hyp:
            print(f"  {hash_id}: 빈 트랜스크립트 — 스킵")
            continue

        w = wer(ref, hyp) * 100  # 퍼센트
        c = cer(ref, hyp) * 100
        row = {
            "hash": hash_id,
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
            "char_delta_pct": round((len(hyp) - len(ref)) / max(1, len(ref)) * 100, 2),
            "wer_pct": round(w, 2),
            "cer_pct": round(c, 2),
        }
        rows.append(row)
        print(
            f"  {hash_id} | WER={w:5.2f}% | CER={c:5.2f}% | "
            f"ref={len(ref):>5}ch | hyp={len(hyp):>5}ch | Δ={row['char_delta_pct']:+.1f}%"
        )

    if not rows:
        print("유효한 비교 결과 없음.")
        return 1

    avg_wer = sum(r["wer_pct"] for r in rows) / len(rows)
    avg_cer = sum(r["cer_pct"] for r in rows) / len(rows)
    max_wer = max(r["wer_pct"] for r in rows)

    summary = {
        "samples": rows,
        "avg_wer_pct": round(avg_wer, 2),
        "avg_cer_pct": round(avg_cer, 2),
        "max_wer_pct": round(max_wer, 2),
        "threshold_pct": args.threshold,
        "verdict": "ACCEPT" if avg_wer <= args.threshold else "REJECT",
        "note": "ground truth 없음 — baseline 기준 turbo diff. WER > threshold 는 수동 검토 필요.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"평균 WER: {avg_wer:.2f}% | 평균 CER: {avg_cer:.2f}% | 최대 WER: {max_wer:.2f}%")
    print(f"임계값: {args.threshold:.1f}% | 판정: {summary['verdict']}")
    print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
