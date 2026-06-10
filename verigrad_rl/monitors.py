"""Experiment monitoring and reward-hacking checks."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List

from verigrad_rl.envs.base import Task, TextEnv


class JsonlLogger:
    """Append-only JSONL logger with config and summary helpers."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def write_config(self, config: Any) -> None:
        payload = asdict(config) if is_dataclass(config) else dict(config)
        (self.run_dir / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log(self, metrics: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class RewardHackingMonitor:
    """Probe verifiers for common false-positive failure modes."""

    def __init__(self, env: TextEnv) -> None:
        self.env = env

    def probe(self, tasks: Iterable[Task]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for task in tasks:
            findings.extend(self._probe_task(task))
        return findings

    def sample_and_probe(self, rng: Random, count: int = 25) -> List[Dict[str, Any]]:
        tasks = [self.env.sample_task(rng, split="train") for _ in range(count)]
        return self.probe(tasks)

    def _probe_task(self, task: Task) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        correct = self.env.verify(task, task.answer)
        if not correct.passed or correct.reward <= 0:
            findings.append(
                {
                    "severity": "critical",
                    "type": "false_negative",
                    "prompt": task.prompt,
                    "answer": task.answer,
                    "reason": correct.reason,
                }
            )

        adversarial_actions = ["", " " + task.answer + " extra", "I think " + task.answer]
        for action in adversarial_actions:
            result = self.env.verify(task, action)
            if result.passed or result.reward >= correct.reward:
                findings.append(
                    {
                        "severity": "high",
                        "type": "false_positive",
                        "prompt": task.prompt,
                        "action": action,
                        "reason": result.reason,
                    }
                )
        return findings
