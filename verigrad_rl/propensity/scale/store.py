"""SQLite results store: caching, provenance, resumability, cost ledger.

Every sample is content-addressed by a cache key over everything that affects the
output (model id, domain, pressure, intensity, problem, sample index, prompt
version). That makes runs **resumable** (kill and restart and finished samples are
reused) and re-runs after a code change only pay for what actually changed. Each row
also carries provenance (model id, prompt version, harness git SHA, seed) so a result
can always be traced back to exactly how it was produced.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def harness_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:  # pragma: no cover
        return "nogit"


def cache_key(model_id: str, domain: str, pressure: str, intensity: int,
              problem_id: str, sample_index: int, prompt_version: str) -> str:
    raw = "|".join([model_id, domain, pressure, str(intensity), problem_id,
                    str(sample_index), prompt_version])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  cache_key TEXT PRIMARY KEY,
  run_id TEXT, domain TEXT, pressure TEXT, intensity INT,
  model TEXT, model_id TEXT, problem_id TEXT, sample_index INT,
  prompt_version TEXT, response TEXT, gold TEXT, answer TEXT, anchor TEXT,
  answered INT, correct INT, deferred INT,
  input_tokens INT, output_tokens INT, cost_usd REAL, latency_s REAL,
  harness_sha TEXT, seed INT, created_at TEXT
);
"""


class BudgetExceeded(RuntimeError):
    pass


class Store:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM samples WHERE cache_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def put(self, record: Dict[str, Any]) -> None:
        cols = ",".join(record)
        ph = ",".join("?" for _ in record)
        self.conn.execute(f"INSERT OR REPLACE INTO samples ({cols}) VALUES ({ph})",
                          list(record.values()))
        self.conn.commit()

    def rows(self, run_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM samples WHERE run_id = ?", (run_id,))
        return [dict(r) for r in cur.fetchall()]

    def total_cost(self) -> float:
        return self.conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM samples").fetchone()[0]

    def export_jsonl(self, run_id: str, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for r in self.rows(run_id):
                fh.write(json.dumps(r) + "\n")
