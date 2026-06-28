"""Command line interface for VeriGrad RL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from verigrad_rl.envs.arithmetic import ArithmeticEnv, ArithmeticSpec
from verigrad_rl.envs.biosafety import BioSafetyEnv
from verigrad_rl.envs.safety_circuit import SafetyCircuitEnv
from verigrad_rl.eval import Evaluator
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


def build_arithmetic_env(args: argparse.Namespace) -> ArithmeticEnv:
    operations = tuple(args.operations.split(","))
    return ArithmeticEnv(
        ArithmeticSpec(
            max_number=args.max_number,
            eval_max_number=args.eval_max_number,
            operations=operations,
        )
    )


def build_env(args: argparse.Namespace):
    if args.env == "arithmetic":
        return build_arithmetic_env(args)
    if args.env == "safety-circuit":
        return SafetyCircuitEnv()
    if args.env == "biosafety":
        return BioSafetyEnv()
    raise ValueError(f"Unknown environment: {args.env}")


def train_command(args: argparse.Namespace) -> None:
    env = build_env(args)
    policy = SoftmaxTextPolicy(env.candidate_actions(), temperature=args.temperature)
    config = TrainingConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        baseline_decay=args.baseline_decay,
        eval_every=args.eval_every,
        eval_tasks=args.eval_tasks,
        seed=args.seed,
        run_dir=args.run_dir,
    )
    summary = Trainer(env, policy, config).train()
    print(json.dumps(summary, indent=2, sort_keys=True))


def eval_command(args: argparse.Namespace) -> None:
    env = build_env(args)
    policy = SoftmaxTextPolicy.load(args.checkpoint)
    report = Evaluator(env, policy, seed=args.seed).run(tasks=args.tasks, split=args.split)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VeriGrad RL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a text policy with verifiable rewards.")
    add_env_args(train)
    train.add_argument("--episodes", type=int, default=1_000)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--baseline-decay", type=float, default=0.95)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--eval-tasks", type=int, default=100)
    train.add_argument("--temperature", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--run-dir", default="runs/latest")
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("eval", help="Evaluate a saved policy checkpoint.")
    add_env_args(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tasks", type=int, default=100)
    evaluate.add_argument("--split", choices=["train", "eval"], default="eval")
    evaluate.add_argument("--seed", type=int, default=13)
    evaluate.set_defaults(func=eval_command)

    propensity = subparsers.add_parser(
        "propensity",
        help="Run the Answer-Under-Pressure benchmark on real frontier models.",
    )
    propensity.add_argument("--models", default="opus-4.8,sonnet-4.6,haiku-4.5")
    propensity.add_argument("--tasks", type=int, default=150)
    propensity.add_argument("--seed", type=int, default=7)
    propensity.add_argument("--workers", type=int, default=6)
    propensity.add_argument("--judge-cap", type=int, default=50)
    propensity.add_argument("--smoke", action="store_true")
    propensity.set_defaults(func=propensity_command)

    scale = subparsers.add_parser(
        "scale",
        help="Run the scalable multi-domain / multi-pressure propensity experiment.",
    )
    scale.add_argument("--domains", default="gsm8k,commonsense_qa")
    scale.add_argument("--models", default="opus-4.8,sonnet-4.6,haiku-4.5")
    scale.add_argument("--tasks", type=int, default=12)
    scale.add_argument("--samples", type=int, default=3, help="samples per item (clustered CIs)")
    scale.add_argument("--intensities", default="1,3", help="authority-pressure levels (1-3)")
    scale.add_argument("--seed", type=int, default=11)
    scale.add_argument("--workers", type=int, default=8)
    scale.add_argument("--budget", type=float, default=3.0, help="hard cost ceiling in USD")
    scale.add_argument("--run-id", default="scale-v1")
    scale.set_defaults(func=scale_command)

    circuit = subparsers.add_parser(
        "circuit",
        help="Automated circuit discovery (ACDC + path patching) on a safety circuit.",
    )
    circuit.add_argument("--target", choices=["safety-dag", "toy-circuit"], default="safety-dag")
    circuit.add_argument("--tau", type=float, default=0.02, help="KL budget per edge (nats)")
    circuit.add_argument("--samples", type=int, default=24, help="contrastive pairs in the task")
    circuit.add_argument("--seed", type=int, default=1)
    circuit.add_argument("--out", default="benchmark/circuits", help="report + SVG output dir")
    circuit.set_defaults(func=circuit_command)
    return parser


def circuit_command(args: argparse.Namespace) -> None:
    from pathlib import Path

    from verigrad_rl.mech.acdc import contrastive_dataset, run_acdc
    from verigrad_rl.mech.circuit_graph import from_toy_safety_circuit, safety_dag
    from verigrad_rl.mech.circuit_report import render_markdown, render_svg

    if args.target == "toy-circuit":
        graph = from_toy_safety_circuit()
        clean = {"harmful_intent": 0.9, "helpful_intent": 0.3, "jailbreak_pressure": 0.9,
                 "refusal_prior": 0.2, "uncertainty": 0.2}
        corrupt = {"harmful_intent": 0.1, "helpful_intent": 0.9, "jailbreak_pressure": 0.1,
                   "refusal_prior": 0.2, "uncertainty": 0.2}
        title = "Discovered safety circuit (RL reward model)"
    else:
        graph = safety_dag()
        clean = {"harm": 0.9, "jailbreak": 0.8, "topic": 0.1, "refusal_cue": 0.3, "noise": 0.5}
        corrupt = {"harm": 0.1, "jailbreak": 0.1, "topic": 0.9, "refusal_cue": 0.3, "noise": 0.5}
        title = "Discovered refuse/answer safety circuit"

    dataset = contrastive_dataset(graph, clean, corrupt, n=args.samples, seed=args.seed)
    result = run_acdc(graph, dataset, tau=args.tau)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(render_markdown(graph, result, title=title), encoding="utf-8")
    (out / "fig_circuit.svg").write_text(render_svg(graph, result, title=title), encoding="utf-8")
    print(json.dumps({
        "target": args.target,
        "edges_kept": result.n_edges,
        "edges_total": result.full_edges,
        "sparsity": round(result.sparsity, 3),
        "faithfulness_kl": round(result.faithfulness_kl, 4),
        "tau": result.tau,
        "circuit": [f"{u}->{v}" for (u, v) in result.circuit_edges],
        "out": str(out),
    }, indent=2))


def scale_command(args: argparse.Namespace) -> None:
    from verigrad_rl.propensity.scale.experiment import ScaleConfig, run_experiment
    from verigrad_rl.propensity.scale.report import render

    intensities = [int(x) for x in args.intensities.split(",") if x.strip()]
    pressure_specs = [("honest", {})] + [("authority_wrong", {"intensity": i}) for i in intensities]
    config = ScaleConfig(
        env_names=[d.strip() for d in args.domains.split(",") if d.strip()],
        pressure_specs=pressure_specs,
        model_keys=[m.strip() for m in args.models.split(",") if m.strip()],
        n_tasks=args.tasks,
        k_samples=args.samples,
        seed=args.seed,
        max_workers=args.workers,
        budget_usd=args.budget,
        run_id=args.run_id,
    )
    run_experiment(config)
    render(config.run_dir)


def propensity_command(args: argparse.Namespace) -> None:
    from verigrad_rl.propensity.config import BenchmarkConfig
    from verigrad_rl.propensity.report import render
    from verigrad_rl.propensity.runner import run_benchmark

    config = BenchmarkConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        n_tasks=3 if args.smoke else args.tasks,
        seed=args.seed,
        max_workers=args.workers,
        judge_cap_per_cell=3 if args.smoke else args.judge_cap,
    )
    run_benchmark(config)
    render()


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", choices=["arithmetic", "biosafety", "safety-circuit"], default="arithmetic")
    parser.add_argument("--max-number", type=int, default=9)
    parser.add_argument("--eval-max-number", type=int, default=14)
    parser.add_argument("--operations", default="+,-", help="Comma-separated operations, e.g. '+,-,*'.")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
