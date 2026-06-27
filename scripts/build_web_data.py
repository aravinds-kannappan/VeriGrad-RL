"""Emit data for the Next.js app from the real runs (reproducible).

Writes:
  web/lib/samples.json   — 648 real cross-domain samples for the in-browser ML
  web/lib/problems.json  — a handful of real GSM8K problems for the live probe demo
  web/public/assets/     — the generated SVG figures

    python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_LIB = ROOT / "lib"
WEB_ASSETS = ROOT / "public" / "assets"


def samples() -> None:
    src = ROOT / "benchmark" / "scale" / "samples.jsonl"
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append([
            int(r["intensity"]),
            1 if r["model"] == "sonnet-4.6" else 0,
            1 if r["model"] == "haiku-4.5" else 0,
            1 if r["domain"] == "commonsense_qa" else 0,
            int(r["deferred"]),
        ])
    payload = {
        "features": ["authority intensity", "is Sonnet 4.6", "is Haiku 4.5", "is CommonsenseQA"],
        "rows": rows,
        "n": len(rows),
        "positives": sum(x[-1] for x in rows),
    }
    (WEB_LIB / "samples.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"  samples.json: {len(rows)} rows, {payload['positives']} positive")


def problems() -> None:
    src = ROOT / "data" / "gsm8k_test.jsonl"
    out = []
    for idx, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        q = row["question"]
        if len(q) > 320:
            continue
        gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        try:
            gold = str(int(float(gold)))
        except ValueError:
            continue
        out.append({"id": idx, "question": q, "gold": gold})
        if len(out) >= 6:
            break
    (WEB_LIB / "problems.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  problems.json: {len(out)} real GSM8K problems")


def figures() -> None:
    src = ROOT / "docs" / "assets"
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    n = 0
    for svg in src.glob("fig_*.svg"):
        shutil.copy(svg, WEB_ASSETS / svg.name)
        n += 1
    print(f"  copied {n} figures to public/assets/")


def main() -> None:
    WEB_LIB.mkdir(parents=True, exist_ok=True)
    print("Building web data from real runs:")
    samples()
    problems()
    figures()


if __name__ == "__main__":
    main()
