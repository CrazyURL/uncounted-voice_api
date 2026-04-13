"""Baseline vs Turbo 트랜스크립트 육안 비교.

하나의 샘플에 대해 baseline과 turbo의 세그먼트를 시간순으로 병합 출력한다.
- [B] / [T] 접두사로 어느 모델 출력인지 구분
- 화자 태그, 시작~끝 타임스탬프 포함
- 말미에 두 전체 텍스트의 unified diff (difflib)

사용법:
    python scripts/bench_compare.py \
        --baseline ~/voice-api-bench/results/baseline \
        --turbo ~/voice-api-bench/results/turbo \
        --hash 1b1666fc \
        --out ~/voice-api-bench/results/compare-1b1666fc.md
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def load_segments(result_path: Path) -> list[dict[str, Any]]:
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", []) or data.get("utterances", [])
    return [
        {
            "start": float(s.get("start") or 0),
            "end": float(s.get("end") or 0),
            "text": (s.get("text") or "").strip(),
            "speaker": s.get("speaker") or "?",
        }
        for s in segs
        if (s.get("text") or "").strip()
    ]


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--turbo", required=True, type=Path)
    parser.add_argument("--hash", required=True, help="비교할 샘플 hash (예: 1b1666fc)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    b_path = args.baseline / f"{args.hash}.result.json"
    t_path = args.turbo / f"{args.hash}.result.json"
    if not b_path.exists() or not t_path.exists():
        print(f"결과 파일 없음: {b_path} / {t_path}")
        return 1

    baseline = load_segments(b_path)
    turbo = load_segments(t_path)

    # 시간순 병합: (start, tag, speaker, text)
    merged: list[tuple[float, str, str, str, float]] = []
    for s in baseline:
        merged.append((s["start"], "B", s["speaker"], s["text"], s["end"]))
    for s in turbo:
        merged.append((s["start"], "T", s["speaker"], s["text"], s["end"]))
    merged.sort(key=lambda x: (x[0], x[1]))

    lines: list[str] = []
    lines.append(f"# Compare: {args.hash}")
    lines.append("")
    lines.append(f"- Baseline segments: {len(baseline)}")
    lines.append(f"- Turbo segments:    {len(turbo)}")
    lines.append("")
    lines.append("## Timeline (B=baseline / T=turbo)")
    lines.append("")
    lines.append("```")
    for start, tag, speaker, text, end in merged:
        lines.append(f"[{tag}] {fmt_time(start)}-{fmt_time(end)} {speaker:>12} | {text}")
    lines.append("```")
    lines.append("")

    # 전체 텍스트 비교
    b_full = " ".join(s["text"] for s in baseline)
    t_full = " ".join(s["text"] for s in turbo)

    lines.append("## Full baseline")
    lines.append("")
    lines.append("```")
    lines.append(b_full)
    lines.append("```")
    lines.append("")
    lines.append("## Full turbo")
    lines.append("")
    lines.append("```")
    lines.append(t_full)
    lines.append("```")
    lines.append("")

    # 문자 단위 unified diff
    lines.append("## Unified diff (char-level, context=2)")
    lines.append("")
    lines.append("```diff")
    diff = difflib.unified_diff(
        b_full.split(),
        t_full.split(),
        fromfile="baseline",
        tofile="turbo",
        n=2,
        lineterm="",
    )
    for d in diff:
        lines.append(d)
    lines.append("```")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {args.out}")
    print(f"baseline: {len(baseline)}개 세그먼트, {len(b_full)}자")
    print(f"turbo:    {len(turbo)}개 세그먼트, {len(t_full)}자")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
