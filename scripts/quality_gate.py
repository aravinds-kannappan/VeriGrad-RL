"""Run the local quality gate without third-party tools."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        [
            sys.executable,
            "-m",
            "verigrad_rl.cli",
            "train",
            "--episodes",
            "50",
            "--eval-every",
            "25",
            "--eval-tasks",
            "20",
            "--run-dir",
            "runs/quality-gate",
        ],
        [
            sys.executable,
            "-m",
            "verigrad_rl.cli",
            "train",
            "--env",
            "safety-circuit",
            "--episodes",
            "120",
            "--eval-every",
            "60",
            "--eval-tasks",
            "30",
            "--temperature",
            "1.5",
            "--run-dir",
            "runs/safety-quality-gate",
        ],
    ]
    for command in commands:
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
