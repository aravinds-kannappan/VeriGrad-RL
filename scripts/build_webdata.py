"""Emit the in-browser ML training data from the real scale run.

Reads benchmark/scale/samples.jsonl (real per-sample outcomes from the
cross-domain run) and writes docs/data.js — a compact feature matrix the site's
in-browser logistic regression trains on. Reproducible:

    python scripts/build_webdata.py

Features (reference levels = opus-4.8, gsm8k): authority intensity, is-sonnet,
is-haiku, is-commonsenseQA. Target: did the model defer to the wrong authority?
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "benchmark" / "scale" / "samples.jsonl"
OUT = ROOT / "docs" / "data.js"


def main() -> None:
    rows = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
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
    positives = sum(row[-1] for row in rows)
    payload = {
        "features": ["authority intensity", "is Sonnet 4.6", "is Haiku 4.5", "is CommonsenseQA"],
        "rows": rows,
        "n": len(rows),
        "positives": positives,
        "source": "benchmark/scale/samples.jsonl (real cross-domain run)",
    }
    OUT.write_text("window.AUP = " + json.dumps(payload) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} · {len(rows)} rows · {positives} positive (deferred) · "
          f"{100*positives/len(rows):.1f}% base rate")


if __name__ == "__main__":
    main()
