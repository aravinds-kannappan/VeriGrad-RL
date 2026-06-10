"""Command line interface for VeriGrad RL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from verigrad_rl.envs.arithmetic import ArithmeticEnv, ArithmeticSpec
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


def train_command(args: argparse.Namespace) -> None:
    env = build_arithmetic_env(args)
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
    env = build_arithmetic_env(args)
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
    return parser


def add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-number", type=int, default=9)
    parser.add_argument("--eval-max-number", type=int, default=14)
    parser.add_argument("--operations", default="+,-", help="Comma-separated operations, e.g. '+,-,*'.")


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
