#!/usr/bin/env python3
"""Voice API Option B (WeSpeaker Recluster) end-to-end 검증.

Phase 4-7 (pyannote 임베딩 → 윈도우 빌더 → AHC 재클러스터링 → stt_processor
훅)의 실오디오 회귀 검증 스크립트. DB 접근 없음, voice-api HTTP만 사용.

흐름:
    1. 로컬 sample_data/Call m4a 4개를 voice-api로 POST
    2. /api/v1/jobs/{task_id}를 폴링해 결과 수집
    3. word-level 화자 분포 집계 후 JSON 저장
       (.validation/recluster_{label}/{id}.json, summary.json)
    4. LABEL=compare로 실행하면 baseline vs option-b 마크다운 리포트 생성

사용법:
    cd uncounted-voice-api

    # 1) 기준 상태(Option D on, recluster off) 측정
    LABEL=baseline python scripts/verify_recluster_option_b.py

    # 2) 서버에 VOICE_DIARIZATION_WESPEAKER_RECLUSTER=true 적용 후
    LABEL=option-b python scripts/verify_recluster_option_b.py

    # 3) 두 결과 비교 리포트
    LABEL=compare python scripts/verify_recluster_option_b.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

VOICE_API = os.environ.get("VOICE_API_URL", "http://183.96.42.95:8001")
LABEL = os.environ.get("LABEL", "baseline")
SAMPLE_DIR = Path(
    os.environ.get(
        "SAMPLE_DIR", "/Users/gdash/project/uncounted-project/sample_data/Call"
    )
)
OUT_ROOT = Path(".validation").resolve()
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 20 * 60

TARGETS: list[dict[str, str]] = [
    {"id": "7b2803baa5f6fa8d", "file": "통화 녹음 김곰박_260403_185311.m4a"},
    {"id": "c8bfc028c5fbb5fd", "file": "통화 녹음 임명훈_260317_210002.m4a"},
    {"id": "a413c7bc6209626b", "file": "통화 녹음 임명훈_260313_193223.m4a"},
    {"id": "71ab5b257f263e0d", "file": "통화 녹음 문식환_260316_212642.m4a"},
]


def _post_transcribe(file_path: Path) -> dict[str, Any]:
    url = f"{VOICE_API}/api/v1/transcribe"
    params = {
        "language": "ko",
        "diarize": "true",
        "mask_pii": "false",
        "split_by_utterance": "true",
        "split_by_speaker": "false",
    }
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "audio/m4a")}
        r = requests.post(url, params=params, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def _poll_job(task_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_SEC
    while time.time() < deadline:
        r = requests.get(f"{VOICE_API}/api/v1/jobs/{task_id}", timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"job failed: {data.get('error', '?')}")
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError("poll timeout")


def _collect_speaker_stats(result: dict[str, Any]) -> dict[str, Any]:
    speaker_word_count: dict[str, int] = {}
    speaker_utt_count: dict[str, int] = {}
    total_words = 0
    total_utts = 0
    for u in result.get("utterances") or []:
        sid = u.get("speaker_id") or u.get("speakerId") or "UNKNOWN"
        speaker_utt_count[sid] = speaker_utt_count.get(sid, 0) + 1
        total_utts += 1
        for w in u.get("words") or []:
            ws = w.get("speaker_id") or w.get("speaker") or sid
            speaker_word_count[ws] = speaker_word_count.get(ws, 0) + 1
            total_words += 1
    return {
        "total_words": total_words,
        "total_utterances": total_utts,
        "speaker_word_count": speaker_word_count,
        "speaker_utterance_count": speaker_utt_count,
        "distinct_speakers": len(speaker_word_count),
    }


def _run_one(target: dict[str, str], run_dir: Path) -> dict[str, Any]:
    file_path = SAMPLE_DIR / target["file"]
    if not file_path.exists():
        raise FileNotFoundError(f"missing: {file_path}")
    started = time.time()
    print(f"[{target['id']}] POST {target['file']}")
    task = _post_transcribe(file_path)
    task_id = task["task_id"]
    print(f"[{target['id']}] task_id={task_id}")
    result = _poll_job(task_id)
    elapsed = time.time() - started
    stats = _collect_speaker_stats(result)
    duration = result.get("audio_duration_sec") or result.get("duration")
    rtf = duration / elapsed if duration else None
    record: dict[str, Any] = {
        "id": target["id"],
        "file": target["file"],
        "label": LABEL,
        "task_id": task_id,
        "elapsed_sec": round(elapsed, 2),
        "audio_duration_sec": duration,
        "rtf": rtf,
        **stats,
    }
    (run_dir / f"{target['id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2)
    )
    rtf_str = f"{rtf:.1f}x" if rtf else "?"
    print(
        f"[{target['id']}] done "
        f"{stats['distinct_speakers']} speakers, {stats['total_words']} words, rtf={rtf_str}"
    )
    return record


def _run_all() -> None:
    run_dir = OUT_ROOT / f"recluster_{LABEL}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for t in TARGETS:
        try:
            records.append(_run_one(t, run_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[{t['id']}] FAIL {e}", file=sys.stderr)
            records.append({"id": t["id"], "file": t["file"], "label": LABEL, "error": str(e)})
    summary = {
        "label": LABEL,
        "voice_api": VOICE_API,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": records,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\nwrote {run_dir / 'summary.json'}")


def _load_summary(label: str) -> dict[str, Any] | None:
    p = OUT_ROOT / f"recluster_{label}" / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _compare() -> None:
    a = _load_summary("baseline")
    b = _load_summary("option-b")
    if not a or not b:
        print("both baseline and option-b summaries required", file=sys.stderr)
        sys.exit(1)
    lines: list[str] = []
    lines.append("# Phase 4-7 Option B Validation Report")
    lines.append(f"generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    lines.append(f"voice-api: {a['voice_api']}")
    lines.append("")
    lines.append(
        "| file | baseline spkrs | option-b spkrs | baseline words | option-b words | "
        "Δ speakers | baseline rtf | option-b rtf |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    b_by_id = {r["id"]: r for r in b["records"]}
    for ar in a["records"]:
        br = b_by_id.get(ar["id"], {})
        a_sp = ar.get("distinct_speakers", "?")
        b_sp = br.get("distinct_speakers", "?")
        a_w = ar.get("total_words", "?")
        b_w = br.get("total_words", "?")
        d_sp = (
            b_sp - a_sp if isinstance(a_sp, int) and isinstance(b_sp, int) else "?"
        )
        a_rtf = f"{ar['rtf']:.1f}x" if ar.get("rtf") else "?"
        b_rtf = f"{br['rtf']:.1f}x" if br.get("rtf") else "?"
        name = ar["file"].replace("통화 녹음 ", "")
        lines.append(
            f"| {name} | {a_sp} | {b_sp} | {a_w} | {b_w} | {d_sp} | {a_rtf} | {b_rtf} |"
        )
    lines.append("")
    lines.append("## Per-file distribution diff")
    for ar in a["records"]:
        br = b_by_id.get(ar["id"], {})
        lines.append(f"\n### {ar['file']}")
        lines.append("```")
        lines.append("baseline word counts: " + json.dumps(ar.get("speaker_word_count", {}), ensure_ascii=False))
        lines.append("option-b word counts: " + json.dumps(br.get("speaker_word_count", {}), ensure_ascii=False))
        lines.append("```")
    out_path = OUT_ROOT / "recluster_report.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print("\n" + "\n".join(lines))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if LABEL == "compare":
        _compare()
    else:
        _run_all()


if __name__ == "__main__":
    main()
