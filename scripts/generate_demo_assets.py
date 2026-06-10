"""Generate static demo assets for the README, notebook, and GitHub Pages site."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verigrad_rl.envs import ArithmeticEnv
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
NOTEBOOKS = ROOT / "notebooks"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    metrics = run_demo_training()
    write_training_curve(metrics)
    write_system_diagram()
    write_reward_pipeline()
    write_notebook(metrics)


def run_demo_training() -> list[dict[str, object]]:
    run_dir = ROOT / "runs" / "pages-demo"
    env = ArithmeticEnv()
    policy = SoftmaxTextPolicy(env.candidate_actions())
    config = TrainingConfig(
        episodes=50,
        eval_every=5,
        eval_tasks=50,
        seed=7,
        run_dir=str(run_dir),
    )
    Trainer(env, policy, config).train()
    rows = []
    with (run_dir / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "train_accuracy" in row:
                rows.append(row)
    return rows


def write_training_curve(metrics: list[dict[str, object]]) -> None:
    width, height = 860, 420
    pad_left, pad_bottom, pad_top, pad_right = 70, 60, 40, 30
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    def point(row: dict[str, object], key: str) -> tuple[float, float]:
        max_episode = max(float(item["episode"]) for item in metrics)
        x = pad_left + (float(row["episode"]) / max_episode) * chart_w
        y = pad_top + (1.0 - float(row[key])) * chart_h
        return x, y

    def path_for(key: str) -> str:
        coords = [point(row, key) for row in metrics]
        return " ".join(("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}" for idx, (x, y) in enumerate(coords))

    train_path = path_for("train_accuracy")
    eval_path = path_for("eval_accuracy")
    grid = "\n".join(
        f'<line x1="{pad_left}" y1="{pad_top + i * chart_h / 4:.1f}" '
        f'x2="{width - pad_right}" y2="{pad_top + i * chart_h / 4:.1f}" class="grid"/>'
        for i in range(5)
    )
    labels = "\n".join(
        f'<text x="30" y="{pad_top + i * chart_h / 4 + 5:.1f}" class="tick">{1 - i / 4:.2f}</text>'
        for i in range(5)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">VeriGrad RL training curve</title>
  <desc id="desc">Train and held-out evaluation accuracy over a 50 episode verifier-guided RL run.</desc>
  <style>
    .bg {{ fill: #fbfaf7; }}
    .axis {{ stroke: #1f2933; stroke-width: 1.5; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; }}
    .train {{ fill: none; stroke: #0f766e; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
    .eval {{ fill: none; stroke: #b45309; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
    .label {{ fill: #1f2933; font: 700 20px system-ui, sans-serif; }}
    .small {{ fill: #52606d; font: 14px system-ui, sans-serif; }}
    .tick {{ fill: #52606d; font: 12px system-ui, sans-serif; text-anchor: end; }}
  </style>
  <rect class="bg" width="{width}" height="{height}" rx="0"/>
  <text x="{pad_left}" y="25" class="label">Training learns exact verifier behavior in 50 episodes</text>
  {grid}
  {labels}
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis"/>
  <path d="{train_path}" class="train"/>
  <path d="{eval_path}" class="eval"/>
  <text x="{width - 230}" y="78" class="small">Train accuracy</text>
  <line x1="{width - 270}" y1="73" x2="{width - 240}" y2="73" class="train"/>
  <text x="{width - 230}" y="104" class="small">Held-out eval accuracy</text>
  <line x1="{width - 270}" y1="99" x2="{width - 240}" y2="99" class="eval"/>
  <text x="{width / 2}" y="{height - 18}" class="small" text-anchor="middle">Episode</text>
  <text x="18" y="{height / 2}" class="small" text-anchor="middle" transform="rotate(-90 18 {height / 2})">Accuracy</text>
</svg>
"""
    (ASSETS / "training_curve.svg").write_text(svg, encoding="utf-8")


def write_system_diagram() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 360" role="img" aria-labelledby="title desc">
  <title id="title">VeriGrad RL system diagram</title>
  <desc id="desc">Task sampling, policy rollout, verifier scoring, training, evals, and monitoring form the core loop.</desc>
  <style>
    .bg { fill: #f8fafc; }
    .box { fill: #ffffff; stroke: #334e68; stroke-width: 2; }
    .accent { fill: #e0f2fe; stroke: #0369a1; }
    .warn { fill: #fef3c7; stroke: #b45309; }
    .text { fill: #102a43; font: 700 18px system-ui, sans-serif; text-anchor: middle; }
    .sub { fill: #52606d; font: 13px system-ui, sans-serif; text-anchor: middle; }
    .arrow { stroke: #334e68; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }
  </style>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#334e68"/></marker></defs>
  <rect class="bg" width="960" height="360"/>
  <rect class="box" x="40" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="115" y="145">Environment</text><text class="sub" x="115" y="170">sample task</text>
  <rect class="box" x="240" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="315" y="145">Policy</text><text class="sub" x="315" y="170">sample action</text>
  <rect class="accent" x="440" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="515" y="145">Verifier</text><text class="sub" x="515" y="170">reward + reason</text>
  <rect class="box" x="640" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="715" y="145">Trainer</text><text class="sub" x="715" y="170">policy gradient</text>
  <rect class="warn" x="340" y="250" width="280" height="70" rx="6"/>
  <text class="text" x="480" y="280">Monitoring + Evals</text><text class="sub" x="480" y="303">accuracy, failures, reward hacking probes</text>
  <path class="arrow" d="M190 152 H238"/>
  <path class="arrow" d="M390 152 H438"/>
  <path class="arrow" d="M590 152 H638"/>
  <path class="arrow" d="M715 105 C715 45 315 45 315 103"/>
  <path class="arrow" d="M515 200 V248"/>
</svg>
"""
    (ASSETS / "system_diagram.svg").write_text(svg, encoding="utf-8")


def write_reward_pipeline() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300" role="img" aria-labelledby="title desc">
  <title id="title">Verifier reward pipeline</title>
  <desc id="desc">A text action is parsed, checked for format, scored for exactness, and logged with diagnostic metadata.</desc>
  <style>
    .bg { fill: #fffdf7; }
    .step { fill: #ffffff; stroke: #475569; stroke-width: 2; }
    .pass { fill: #dcfce7; stroke: #15803d; }
    .fail { fill: #fee2e2; stroke: #b91c1c; }
    .text { fill: #172554; font: 700 17px system-ui, sans-serif; text-anchor: middle; }
    .sub { fill: #475569; font: 13px system-ui, sans-serif; text-anchor: middle; }
    .arrow { stroke: #475569; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }
  </style>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#475569"/></marker></defs>
  <rect class="bg" width="900" height="300"/>
  <rect class="step" x="40" y="90" width="145" height="90" rx="6"/>
  <text class="text" x="112" y="128">Text action</text><text class="sub" x="112" y="153">"14"</text>
  <rect class="step" x="235" y="90" width="145" height="90" rx="6"/>
  <text class="text" x="307" y="128">Format check</text><text class="sub" x="307" y="153">integer only</text>
  <rect class="step" x="430" y="90" width="145" height="90" rx="6"/>
  <text class="text" x="502" y="128">Exact check</text><text class="sub" x="502" y="153">compare answer</text>
  <rect class="pass" x="625" y="45" width="210" height="75" rx="6"/>
  <text class="text" x="730" y="75">Pass</text><text class="sub" x="730" y="99">reward = 1.0, reason = exact_match</text>
  <rect class="fail" x="625" y="165" width="210" height="75" rx="6"/>
  <text class="text" x="730" y="195">Fail</text><text class="sub" x="730" y="219">shaped reward + diagnostics</text>
  <path class="arrow" d="M185 135 H233"/>
  <path class="arrow" d="M380 135 H428"/>
  <path class="arrow" d="M575 125 C595 105 600 90 623 83"/>
  <path class="arrow" d="M575 145 C595 165 600 195 623 202"/>
</svg>
"""
    (ASSETS / "reward_pipeline.svg").write_text(svg, encoding="utf-8")


def write_notebook(metrics: list[dict[str, object]]) -> None:
    final = metrics[-1]
    cells = [
        markdown(
            "# VeriGrad RL Walkthrough\n\n"
            "This notebook is a GitHub-renderable tour of VeriGrad RL: a compact "
            "verifiable-reward RL lab for text agents. It shows the motivation, "
            "the system architecture, the reward pipeline, and a reproducible "
            "training run."
        ),
        markdown("## System loop\n\n![System diagram](../docs/assets/system_diagram.svg)"),
        markdown("## Reward pipeline\n\n![Reward pipeline](../docs/assets/reward_pipeline.svg)"),
        code(
            "from verigrad_rl.envs import ArithmeticEnv\n"
            "from verigrad_rl.policy import SoftmaxTextPolicy\n"
            "from verigrad_rl.train import Trainer, TrainingConfig\n\n"
            "env = ArithmeticEnv()\n"
            "policy = SoftmaxTextPolicy(env.candidate_actions())\n"
            "config = TrainingConfig(episodes=50, eval_every=5, eval_tasks=50, run_dir='runs/notebook-demo')\n"
            "summary = Trainer(env, policy, config).train()\n"
            "summary"
        ),
        markdown(
            "## Training curve\n\n"
            "The default run trains from verifier feedback and evaluates greedily "
            "on both train and held-out arithmetic splits.\n\n"
            "![Training curve](../docs/assets/training_curve.svg)"
        ),
        code(
            "from verigrad_rl.eval import Evaluator\n\n"
            "report = Evaluator(env, policy, seed=123).run(tasks=100, split='eval')\n"
            "report"
        ),
        markdown(
            "## Final generated demo metrics\n\n"
            f"- Train accuracy: `{final['train_accuracy']}`\n"
            f"- Held-out eval accuracy: `{final['eval_accuracy']}`\n"
            "- Reward-hacking probes: verifier rejects blank, extra-text, and "
            "non-integer outputs while accepting exact answers.\n\n"
            "The arithmetic task is deliberately simple; the value is the reusable "
            "infrastructure shape: tasks, rollouts, verifiers, trainers, evals, "
            "logs, and reward-hacking checks."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.9+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "VeriGrad_RL_walkthrough.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


if __name__ == "__main__":
    main()
