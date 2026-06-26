"""Configuration: env loading, model registry, pricing, and paths.

Pricing is list price per 1M tokens as published for these model IDs. Token
counts in the report are *measured* from real API usage; dollar figures are
those measured tokens multiplied by the rates below. Update RATES if list
prices change -- the harness never invents a number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Repo root (…/VeriGrad-RL). config.py lives at verigrad_rl/propensity/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "benchmark" / "results"


@dataclass(frozen=True)
class ModelSpec:
    """A real model the benchmark can evaluate."""

    key: str  # short label used in the leaderboard
    model_id: str  # exact Anthropic API model string
    input_per_mtok: float  # USD per 1M input tokens (list price)
    output_per_mtok: float  # USD per 1M output tokens (list price)


# Anthropic lineup. Model IDs and list prices are authoritative as of 2026-06.
MODELS: Dict[str, ModelSpec] = {
    "opus-4.8": ModelSpec("opus-4.8", "claude-opus-4-8", 5.00, 25.00),
    "sonnet-4.6": ModelSpec("sonnet-4.6", "claude-sonnet-4-6", 3.00, 15.00),
    "haiku-4.5": ModelSpec("haiku-4.5", "claude-haiku-4-5", 1.00, 5.00),
}

# Independent grader model used only for the reliability cross-check.
JUDGE_MODEL_ID = "claude-haiku-4-5"

CONDITIONS = ("control", "authority_wrong", "grader_gameable")


@dataclass
class BenchmarkConfig:
    """Knobs for one benchmark run."""

    models: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    conditions: List[str] = field(default_factory=lambda: list(CONDITIONS))
    n_tasks: int = 150
    seed: int = 7
    max_workers: int = 6
    max_tokens: int = 1024
    judge_cap_per_cell: int = 50  # how many records per (model,condition) to dual-label
    run_dir: Path = RESULTS_DIR


def load_dotenv(path: Path | None = None) -> Dict[str, str]:
    """Minimal .env loader (no dependency). Returns the keys it set.

    Only sets variables that are not already present in the environment, so an
    exported key always wins over the file.
    """

    env_path = path or (REPO_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
