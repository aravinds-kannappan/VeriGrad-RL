"""Generate static demo assets for the README, notebook, and GitHub Pages site."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verigrad_rl.envs import ArithmeticEnv, SafetyCircuitEnv
from verigrad_rl.eval import Evaluator
from verigrad_rl.mech import ActivationSnapshot
from verigrad_rl.monitors import RewardHackingMonitor
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
NOTEBOOKS = ROOT / "notebooks"
PROFILE_ORDER = ["benign", "ambiguous", "harmful", "jailbreak", "benign_ood", "jailbreak_ood"]


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    arithmetic_metrics = run_arithmetic_training()
    demo = build_safety_demo()

    write_training_curve(arithmetic_metrics)
    write_safety_dashboard(demo["metrics"])
    write_feature_attribution(demo["attribution"])
    write_intervention_comparison(demo["intervention_rows"])
    write_behavior_logits(demo["before_logits"], demo["after_logits"])
    write_system_diagram()
    write_safety_pipeline()
    write_notebook(arithmetic_metrics, demo)


def run_arithmetic_training() -> List[Dict[str, Any]]:
    run_dir = ROOT / "runs" / "pages-arithmetic-demo"
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
    return read_metric_rows(run_dir / "metrics.jsonl", required_key="train_accuracy")


def build_safety_demo() -> Dict[str, Any]:
    run_dir = ROOT / "runs" / "pages-safety-demo"
    env = SafetyCircuitEnv()
    policy = SoftmaxTextPolicy(env.candidate_actions(), temperature=1.5)
    config = TrainingConfig(
        episodes=3_000,
        eval_every=300,
        eval_tasks=200,
        learning_rate=0.12,
        seed=7,
        run_dir=str(run_dir),
    )
    summary = Trainer(env, policy, config).train()
    metrics = read_metric_rows(run_dir / "metrics.jsonl", required_key="eval_extra_metrics")

    tasks = {
        profile: find_profile_task(env, profile, split="eval" if profile.endswith("_ood") else "train")
        for profile in PROFILE_ORDER
    }
    jailbreak_task = tasks["jailbreak"]
    before = ActivationSnapshot(dict(jailbreak_task.metadata["activation_snapshot"]))
    reference_action = str(jailbreak_task.answer)
    after = env.circuit.apply(before, reference_action)

    intervention_rows = intervention_table(env, jailbreak_task)
    top_action_rows = top_action_table(env, policy, tasks)
    eval_report = Evaluator(env, policy, seed=123).run(tasks=200, split="eval")
    hacking_findings = RewardHackingMonitor(env).sample_and_probe(Random(123), count=25)

    return {
        "env": env,
        "policy": policy,
        "config": config,
        "summary": summary,
        "metrics": metrics,
        "tasks": tasks,
        "jailbreak_task": jailbreak_task,
        "before_logits": env.circuit.logits(before),
        "after_logits": env.circuit.logits(after),
        "attribution": env.circuit.attribution(before),
        "intervention_rows": intervention_rows,
        "top_action_rows": top_action_rows,
        "eval_report": eval_report,
        "hacking_findings": hacking_findings,
    }


def read_metric_rows(path: Path, required_key: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if required_key in row:
                rows.append(row)
    return rows


def find_profile_task(env: SafetyCircuitEnv, profile: str, split: str) -> Any:
    for seed in range(10_000):
        task = env.sample_task(Random(seed), split=split)
        if task.metadata["profile"] == profile:
            return task
    raise RuntimeError(f"Could not sample profile {profile}")


def intervention_table(env: SafetyCircuitEnv, task: Any) -> List[Dict[str, Any]]:
    rows = []
    for action in env.candidate_actions():
        result = env.verify(task, action)
        rows.append(
            {
                "action": action,
                "reward": round(result.reward, 3),
                "passed": result.passed,
                "reason": result.reason,
                "behavior": result.details["after_behavior"],
                "mechanistic_score": round(float(result.details["mechanistic_score"]), 3),
                "safety_score": round(float(result.details["safety_score"]), 3),
                "utility_score": round(float(result.details["utility_score"]), 3),
            }
        )
    return sorted(rows, key=lambda row: float(row["reward"]), reverse=True)


def top_action_table(
    env: SafetyCircuitEnv,
    policy: SoftmaxTextPolicy,
    tasks: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for profile in PROFILE_ORDER:
        task = tasks[profile]
        top = policy.top_actions(task.prompt, k=3)
        action = top[0][0]
        result = env.verify(task, action)
        rows.append(
            {
                "profile": profile,
                "top_action": action,
                "probability": round(top[0][1], 3),
                "behavior": result.details["after_behavior"],
                "passed": result.passed,
                "reward": round(result.reward, 3),
            }
        )
    return rows


def write_training_curve(metrics: List[Dict[str, Any]]) -> None:
    width, height = 860, 420
    pad_left, pad_bottom, pad_top, pad_right = 70, 60, 40, 30
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    def point(row: Dict[str, Any], key: str) -> Tuple[float, float]:
        max_episode = max(float(item["episode"]) for item in metrics)
        x = pad_left + (float(row["episode"]) / max_episode) * chart_w
        y = pad_top + (1.0 - float(row[key])) * chart_h
        return x, y

    def path_for(key: str) -> str:
        coords = [point(row, key) for row in metrics]
        return " ".join(("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}" for idx, (x, y) in enumerate(coords))

    svg = line_chart_svg(
        width=width,
        height=height,
        title="Arithmetic smoke test: verifier behavior in 50 episodes",
        y_label="Accuracy",
        series=[
            ("Train accuracy", "#0f766e", path_for("train_accuracy")),
            ("Held-out eval accuracy", "#b45309", path_for("eval_accuracy")),
        ],
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
    (ASSETS / "training_curve.svg").write_text(svg, encoding="utf-8")


def write_safety_dashboard(metrics: List[Dict[str, Any]]) -> None:
    width, height = 920, 470
    pad_left, pad_bottom, pad_top, pad_right = 78, 62, 50, 34
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    series_specs = [
        ("safety_rate", "Safety", "#0f766e"),
        ("utility_rate", "Utility", "#0369a1"),
        ("mechanistic_alignment_rate", "Mechanistic alignment", "#7c3aed"),
        ("over_refusal_rate", "Over-refusal", "#b45309"),
        ("jailbreak_success_rate", "Jailbreak success", "#b91c1c"),
    ]

    def value(row: Dict[str, Any], key: str) -> float:
        return float(row["eval_extra_metrics"][key])

    def point(row: Dict[str, Any], key: str) -> Tuple[float, float]:
        max_episode = max(float(item["episode"]) for item in metrics)
        x = pad_left + (float(row["episode"]) / max_episode) * chart_w
        y = pad_top + (1.0 - value(row, key)) * chart_h
        return x, y

    series = []
    for key, label, color in series_specs:
        coords = [point(row, key) for row in metrics]
        path = " ".join(
            ("M" if point_idx == 0 else "L") + f"{x:.1f},{y:.1f}"
            for point_idx, (x, y) in enumerate(coords)
        )
        series.append((label, color, path))

    svg = line_chart_svg(
        width=width,
        height=height,
        title="Safety-circuit RL learns surgical activation interventions",
        y_label="Rate",
        series=series,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        legend_x=650,
    )
    (ASSETS / "safety_dashboard.svg").write_text(svg, encoding="utf-8")


def write_feature_attribution(attribution: Dict[str, float]) -> None:
    rows = sorted(attribution.items(), key=lambda item: item[1], reverse=True)
    svg = horizontal_bar_svg(
        title="Causal attribution for unsafe-completion logit",
        rows=[(label, value, "#7c3aed") for label, value in rows],
        value_format="{:.2f}",
    )
    (ASSETS / "feature_attribution.svg").write_text(svg, encoding="utf-8")


def write_intervention_comparison(rows: List[Dict[str, Any]]) -> None:
    chart_rows = [
        (short_action(str(row["action"])), float(row["reward"]), "#0f766e" if row["passed"] else "#b45309")
        for row in rows[:7]
    ]
    svg = horizontal_bar_svg(
        title="Intervention reward on a jailbreak prompt",
        rows=chart_rows,
        value_format="{:.3f}",
    )
    (ASSETS / "intervention_comparison.svg").write_text(svg, encoding="utf-8")


def write_behavior_logits(before_logits: Dict[str, float], after_logits: Dict[str, float]) -> None:
    labels = list(before_logits)
    width, height = 920, 420
    left, top = 90, 70
    bar_w = 30
    group_gap = 120
    max_abs = max(abs(value) for value in list(before_logits.values()) + list(after_logits.values()))
    scale = 120 / max(max_abs, 0.1)
    axis_y = 240
    bars = []
    for idx, label in enumerate(labels):
        x = left + idx * group_gap
        before_h = before_logits[label] * scale
        after_h = after_logits[label] * scale
        bars.append(bar_rect(x, axis_y, bar_w, before_h, "#b45309"))
        bars.append(bar_rect(x + 38, axis_y, bar_w, after_h, "#0f766e"))
        bars.append(f'<text x="{x + 34}" y="305" class="tick" text-anchor="middle">{label}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Behavior logits before and after intervention</title>
  <desc id="desc">Grouped bars compare behavior logits before and after a targeted safety intervention.</desc>
  {svg_style()}
  <rect class="bg" width="{width}" height="{height}"/>
  <text x="70" y="34" class="label">Behavior logits before vs. after targeted intervention</text>
  <line x1="65" y1="{axis_y}" x2="840" y2="{axis_y}" class="axis"/>
  {"".join(bars)}
  <rect x="650" y="76" width="18" height="18" fill="#b45309"/><text x="676" y="91" class="small">Before patch</text>
  <rect x="650" y="106" width="18" height="18" fill="#0f766e"/><text x="676" y="121" class="small">After patch</text>
</svg>
"""
    (ASSETS / "behavior_logits.svg").write_text(svg, encoding="utf-8")


def line_chart_svg(
    width: int,
    height: int,
    title: str,
    y_label: str,
    series: List[Tuple[str, str, str]],
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
    legend_x: int | None = None,
) -> str:
    chart_h = height - pad_top - pad_bottom
    legend_x = legend_x or width - 270
    grid = "\n".join(
        f'<line x1="{pad_left}" y1="{pad_top + i * chart_h / 4:.1f}" '
        f'x2="{width - pad_right}" y2="{pad_top + i * chart_h / 4:.1f}" class="grid"/>'
        for i in range(5)
    )
    labels = "\n".join(
        f'<text x="{pad_left - 34}" y="{pad_top + i * chart_h / 4 + 5:.1f}" class="tick">{1 - i / 4:.2f}</text>'
        for i in range(5)
    )
    paths = []
    legend = []
    for idx, (label, color, path) in enumerate(series):
        paths.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        y = 78 + idx * 24
        legend.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        legend.append(f'<text x="{legend_x + 42}" y="{y + 5}" class="small">{label}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">Line chart generated from VeriGrad RL demo metrics.</desc>
  {svg_style()}
  <rect class="bg" width="{width}" height="{height}"/>
  <text x="{pad_left}" y="30" class="label">{title}</text>
  {grid}
  {labels}
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" class="axis"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" class="axis"/>
  {"".join(paths)}
  {"".join(legend)}
  <text x="{width / 2}" y="{height - 18}" class="small" text-anchor="middle">Episode</text>
  <text x="20" y="{height / 2}" class="small" text-anchor="middle" transform="rotate(-90 20 {height / 2})">{y_label}</text>
</svg>
"""


def horizontal_bar_svg(
    title: str,
    rows: List[Tuple[str, float, str]],
    value_format: str,
    width: int = 920,
) -> str:
    row_h = 42
    height = 94 + row_h * len(rows)
    label_w = 250
    bar_w = width - label_w - 120
    max_value = max([value for _, value, _ in rows] + [1.0])
    pieces = []
    for idx, (label, value, color) in enumerate(rows):
        y = 68 + idx * row_h
        w = max(2, (value / max_value) * bar_w)
        pieces.append(f'<text x="28" y="{y + 18}" class="small">{label}</text>')
        pieces.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="24" fill="{color}" rx="3"/>')
        pieces.append(f'<text x="{label_w + w + 10:.1f}" y="{y + 18}" class="small">{value_format.format(value)}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">Horizontal bar chart generated from VeriGrad RL demo data.</desc>
  {svg_style()}
  <rect class="bg" width="{width}" height="{height}"/>
  <text x="28" y="34" class="label">{title}</text>
  {"".join(pieces)}
</svg>
"""


def bar_rect(x: float, axis_y: float, width: float, signed_height: float, color: str) -> str:
    if signed_height >= 0:
        return f'<rect x="{x}" y="{axis_y - signed_height:.1f}" width="{width}" height="{signed_height:.1f}" fill="{color}" rx="3"/>'
    return f'<rect x="{x}" y="{axis_y}" width="{width}" height="{abs(signed_height):.1f}" fill="{color}" rx="3"/>'


def svg_style() -> str:
    return """<style>
    .bg { fill: #fbfaf7; }
    .axis { stroke: #1f2933; stroke-width: 1.5; }
    .grid { stroke: #d9e2ec; stroke-width: 1; }
    .label { fill: #1f2933; font: 700 20px system-ui, sans-serif; }
    .small { fill: #52606d; font: 14px system-ui, sans-serif; }
    .tick { fill: #52606d; font: 12px system-ui, sans-serif; }
  </style>"""


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
  <text class="text" x="315" y="145">Policy</text><text class="sub" x="315" y="170">choose intervention</text>
  <rect class="accent" x="440" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="515" y="145">Circuit</text><text class="sub" x="515" y="170">activation patch</text>
  <rect class="box" x="640" y="105" width="150" height="95" rx="6"/>
  <text class="text" x="715" y="145">Trainer</text><text class="sub" x="715" y="170">policy gradient</text>
  <rect class="warn" x="340" y="250" width="280" height="70" rx="6"/>
  <text class="text" x="480" y="280">Verifier + Evals</text><text class="sub" x="480" y="303">safety, utility, causal faithfulness</text>
  <path class="arrow" d="M190 152 H238"/>
  <path class="arrow" d="M390 152 H438"/>
  <path class="arrow" d="M590 152 H638"/>
  <path class="arrow" d="M715 105 C715 45 315 45 315 103"/>
  <path class="arrow" d="M515 200 V248"/>
</svg>
"""
    (ASSETS / "system_diagram.svg").write_text(svg, encoding="utf-8")


def write_safety_pipeline() -> None:
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300" role="img" aria-labelledby="title desc">
  <title id="title">Mechanistic safety verifier pipeline</title>
  <desc id="desc">Residual-stream features are patched by an intervention and scored for safety, utility, and mechanistic faithfulness.</desc>
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
  <text class="text" x="112" y="128">Residual stream</text><text class="sub" x="112" y="153">named features</text>
  <rect class="step" x="235" y="90" width="145" height="90" rx="6"/>
  <text class="text" x="307" y="128">Intervention</text><text class="sub" x="307" y="153">activation patch</text>
  <rect class="step" x="430" y="90" width="145" height="90" rx="6"/>
  <text class="text" x="502" y="128">Causal audit</text><text class="sub" x="502" y="153">targeted features</text>
  <rect class="pass" x="625" y="45" width="210" height="75" rx="6"/>
  <text class="text" x="730" y="75">Pass</text><text class="sub" x="730" y="99">safe, useful, faithful</text>
  <rect class="fail" x="625" y="165" width="210" height="75" rx="6"/>
  <text class="text" x="730" y="195">Fail</text><text class="sub" x="730" y="219">unsafe or over-broad</text>
  <path class="arrow" d="M185 135 H233"/>
  <path class="arrow" d="M380 135 H428"/>
  <path class="arrow" d="M575 125 C595 105 600 90 623 83"/>
  <path class="arrow" d="M575 145 C595 165 600 195 623 202"/>
</svg>
"""
    (ASSETS / "reward_pipeline.svg").write_text(svg, encoding="utf-8")


def write_notebook(arithmetic_metrics: List[Dict[str, Any]], demo: Dict[str, Any]) -> None:
    arithmetic_final = arithmetic_metrics[-1]
    safety_final = demo["metrics"][-1]
    safety_extra = safety_final["eval_extra_metrics"]
    summary = demo["summary"]
    eval_report = demo["eval_report"]
    jailbreak_task = demo["jailbreak_task"]
    sample_features = jailbreak_task.metadata["activation_snapshot"]

    cells = [
        markdown(
            "# VeriGrad RL: Mechanistic Safety Walkthrough\n\n"
            "This notebook is a code-heavy, GitHub-renderable tour of VeriGrad RL. "
            "The main demo trains a policy to choose activation-level interventions "
            "inside a synthetic residual-stream safety circuit, then audits safety, "
            "utility, jailbreak robustness, over-refusal, and mechanistic faithfulness."
        ),
        markdown("## 1. System Loop\n\n![System diagram](../docs/assets/system_diagram.svg)"),
        markdown("## 2. Imports and Environment Setup"),
        code(
            "from dataclasses import asdict\n"
            "from random import Random\n\n"
            "from verigrad_rl.envs import SafetyCircuitEnv\n"
            "from verigrad_rl.eval import Evaluator\n"
            "from verigrad_rl.mech import ActivationSnapshot\n"
            "from verigrad_rl.monitors import RewardHackingMonitor\n"
            "from verigrad_rl.policy import SoftmaxTextPolicy\n"
            "from verigrad_rl.train import Trainer, TrainingConfig\n\n"
            "env = SafetyCircuitEnv()\n"
            "env.candidate_actions()",
            result=demo["env"].candidate_actions(),
        ),
        markdown("## 3. Inspect a Safety-Circuit Task\n\nA task includes a user-facing prompt plus hidden residual-stream features used by the verifier."),
        code(
            "task = env.sample_task(Random(2), split='train')\n"
            "{\n"
            "    'prompt': task.prompt,\n"
            "    'reference_action': task.answer,\n"
            "    'metadata': task.metadata,\n"
            "}",
            result={
                "prompt": jailbreak_task.prompt,
                "reference_action": jailbreak_task.answer,
                "activation_snapshot": sample_features,
                "risk_type": jailbreak_task.metadata["risk_type"],
            },
        ),
        markdown("## 4. Mechanistic Reward Pipeline\n\n![Reward pipeline](../docs/assets/reward_pipeline.svg)"),
        markdown("## 5. Causal Attribution\n\nThe toy circuit estimates which residual-stream features causally support unsafe completions by ablating features and measuring logit drops."),
        code(
            "snapshot = ActivationSnapshot(dict(task.metadata['activation_snapshot']))\n"
            "attribution = env.circuit.attribution(snapshot)\n"
            "dict(sorted(attribution.items(), key=lambda item: item[1], reverse=True))",
            result=dict(sorted(demo["attribution"].items(), key=lambda item: item[1], reverse=True)),
        ),
        markdown("![Feature attribution](../docs/assets/feature_attribution.svg)"),
        markdown("## 6. Compare Candidate Interventions\n\nA useful safety intervention is not just safe. It should be targeted, sparse, and preserve helpful features."),
        code(
            "rows = []\n"
            "for action in env.candidate_actions():\n"
            "    result = env.verify(task, action)\n"
            "    rows.append({\n"
            "        'action': action,\n"
            "        'reward': round(result.reward, 3),\n"
            "        'passed': result.passed,\n"
            "        'reason': result.reason,\n"
            "        'behavior': result.details['after_behavior'],\n"
            "        'mechanistic_score': round(result.details['mechanistic_score'], 3),\n"
            "    })\n"
            "sorted(rows, key=lambda row: row['reward'], reverse=True)",
            result=demo["intervention_rows"],
        ),
        markdown("![Intervention comparison](../docs/assets/intervention_comparison.svg)"),
        markdown("## 7. Behavior Logits Before and After a Patch"),
        code(
            "reference = task.answer\n"
            "before = ActivationSnapshot(dict(task.metadata['activation_snapshot']))\n"
            "after = env.circuit.apply(before, reference)\n"
            "{\n"
            "    'reference_action': reference,\n"
            "    'before_logits': env.circuit.logits(before),\n"
            "    'after_logits': env.circuit.logits(after),\n"
            "    'before_behavior': env.circuit.behavior(before),\n"
            "    'after_behavior': env.circuit.behavior(after),\n"
            "}",
            result={
                "reference_action": jailbreak_task.answer,
                "before_logits": demo["before_logits"],
                "after_logits": demo["after_logits"],
                "before_behavior": demo["env"].circuit.behavior(ActivationSnapshot(dict(sample_features))),
                "after_behavior": demo["env"].verify(jailbreak_task, jailbreak_task.answer).details["after_behavior"],
            },
        ),
        markdown("![Behavior logits](../docs/assets/behavior_logits.svg)"),
        markdown("## 8. Train the Intervention Policy"),
        code(
            "policy = SoftmaxTextPolicy(env.candidate_actions(), temperature=1.5)\n"
            "config = TrainingConfig(\n"
            "    episodes=3000,\n"
            "    eval_every=300,\n"
            "    eval_tasks=200,\n"
            "    learning_rate=0.12,\n"
            "    run_dir='runs/notebook-safety-demo',\n"
            ")\n"
            "summary = Trainer(env, policy, config).train()\n"
            "summary",
            result=summary,
        ),
        markdown("## 9. Safety Dashboard\n\n![Safety dashboard](../docs/assets/safety_dashboard.svg)"),
        markdown("## 10. Evaluate OOD Safety Behavior"),
        code(
            "report = Evaluator(env, policy, seed=123).run(tasks=200, split='eval')\n"
            "asdict(report)",
            result={
                "env_name": eval_report.env_name,
                "split": eval_report.split,
                "tasks": eval_report.tasks,
                "accuracy": eval_report.accuracy,
                "mean_reward": eval_report.mean_reward,
                "failures_by_reason": eval_report.failures_by_reason,
                "reward_hacking_findings": eval_report.reward_hacking_findings,
                "extra_metrics": eval_report.extra_metrics,
            },
        ),
        markdown("## 11. Learned Policy by Prompt Profile"),
        code(
            "profiles = ['benign', 'ambiguous', 'harmful', 'jailbreak', 'benign_ood', 'jailbreak_ood']\n"
            "profile_rows = []\n"
            "for profile in profiles:\n"
            "    split = 'eval' if profile.endswith('_ood') else 'train'\n"
            "    task = next(env.sample_task(Random(seed), split=split) for seed in range(10000) if env.sample_task(Random(seed), split=split).metadata['profile'] == profile)\n"
            "    action, probability = policy.top_actions(task.prompt, k=1)[0]\n"
            "    result = env.verify(task, action)\n"
            "    profile_rows.append({\n"
            "        'profile': profile,\n"
            "        'top_action': action,\n"
            "        'probability': round(probability, 3),\n"
            "        'behavior': result.details['after_behavior'],\n"
            "        'passed': result.passed,\n"
            "    })\n"
            "profile_rows",
            result=demo["top_action_rows"],
        ),
        markdown("## 12. Reward-Hacking Probe"),
        code(
            "findings = RewardHackingMonitor(env).sample_and_probe(Random(123), count=25)\n"
            "findings",
            result=demo["hacking_findings"],
        ),
        markdown(
            "## 13. Final Generated Metrics\n\n"
            f"- Safety rate: `{safety_extra['safety_rate']}`\n"
            f"- Utility rate: `{safety_extra['utility_rate']}`\n"
            f"- Mechanistic alignment: `{safety_extra['mechanistic_alignment_rate']}`\n"
            f"- Over-refusal rate: `{safety_extra['over_refusal_rate']}`\n"
            f"- Jailbreak success rate: `{safety_extra['jailbreak_success_rate']}`\n"
            f"- Arithmetic smoke-test eval accuracy: `{arithmetic_final['eval_accuracy']}`\n\n"
            "The key safety idea is that behavior alone is insufficient. VeriGrad RL "
            "also asks whether the intervention targets the right causal features "
            "and avoids unnecessary damage to helpful behavior."
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
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "VeriGrad_RL_walkthrough.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


def markdown(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str, result: Any | None = None) -> Dict[str, Any]:
    outputs = []
    if result is not None:
        outputs.append(
            {
                "data": {"text/plain": pretty(result).splitlines(keepends=True)},
                "execution_count": None,
                "metadata": {},
                "output_type": "execute_result",
            }
        )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    }


def pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def short_action(action: str) -> str:
    return (
        action.replace("steer:", "")
        .replace("jailbreak_detector", "jailbreak")
        .replace("preserve_helpful", "preserve")
        .replace("ask_clarifying", "clarify")
    )


if __name__ == "__main__":
    main()
