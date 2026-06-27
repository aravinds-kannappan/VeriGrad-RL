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
    return parser


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
