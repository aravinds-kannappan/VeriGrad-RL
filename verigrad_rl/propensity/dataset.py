"""Real task source: the public GSM8K test split.

GSM8K is a dataset of human-written grade-school math word problems with
ground-truth answers (Cobbe et al., 2021). We download the official test split
once, cache it, and parse the gold integer answer from the `#### N` marker.
No problems are generated or modified.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import List

import certifi

from verigrad_rl.propensity.config import DATA_DIR

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
CACHE_PATH = DATA_DIR / "gsm8k_test.jsonl"


@dataclass(frozen=True)
class Problem:
    """One real GSM8K problem."""

    id: int
    question: str
    gold: int


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(GSM8K_TEST_URL, headers={"User-Agent": "verigrad-rl"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _parse_gold(answer_field: str) -> int:
    # GSM8K marks the final answer after a '####' delimiter, e.g. "...\n#### 18".
    marker = answer_field.rsplit("####", 1)[-1]
    digits = marker.strip().replace(",", "").replace("$", "")
    return int(float(digits))


def load_problems(limit: int | None = None, *, refresh: bool = False) -> List[Problem]:
    """Load (downloading + caching if needed) the GSM8K test problems."""

    if refresh or not CACHE_PATH.exists():
        _download(CACHE_PATH)

    problems: List[Problem] = []
    for idx, line in enumerate(CACHE_PATH.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        try:
            gold = _parse_gold(row["answer"])
        except (ValueError, KeyError):
            continue
        problems.append(Problem(id=idx, question=row["question"], gold=gold))
        if limit is not None and len(problems) >= limit:
            break
    return problems


def sample_problems(n: int, seed: int) -> List[Problem]:
    """Deterministically sample a paired problem set shared across all models."""

    pool = load_problems()
    rng = Random(seed)
    if n >= len(pool):
        return pool
    return rng.sample(pool, n)
