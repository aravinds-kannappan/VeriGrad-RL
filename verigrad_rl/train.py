"""Training loop for dependency-free policy-gradient experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Dict, List

from verigrad_rl.envs.base import TextEnv
from verigrad_rl.eval import Evaluator
from verigrad_rl.monitors import JsonlLogger, RewardHackingMonitor
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.rollout import Transition, collect_one


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1_000
    learning_rate: float = 0.08
    baseline_decay: float = 0.95
    eval_every: int = 100
    eval_tasks: int = 100
    seed: int = 7
    run_dir: str = "runs/latest"


class Trainer:
    """REINFORCE trainer with a moving reward baseline."""

    def __init__(self, env: TextEnv, policy: SoftmaxTextPolicy, config: TrainingConfig) -> None:
        self.env = env
        self.policy = policy
        self.config = config
        self.rng = Random(config.seed)
        self.logger = JsonlLogger(config.run_dir)

    def train(self) -> Dict[str, Any]:
        self.logger.write_config(self.config)
        baseline = 0.0
        recent: List[Transition] = []

        hacking_findings = RewardHackingMonitor(self.env).sample_and_probe(self.rng)
        self.logger.log({"episode": 0, "reward_hacking_findings": len(hacking_findings)})

        for episode in range(1, self.config.episodes + 1):
            transition = collect_one(self.env, self.policy, self.rng, split="train", greedy=False)
            baseline = (
                self.config.baseline_decay * baseline
                + (1.0 - self.config.baseline_decay) * transition.reward
            )
            advantage = transition.reward - baseline
            self.policy.update(
                transition.task.prompt,
                transition.action,
                advantage,
                self.config.learning_rate,
            )
            recent.append(transition)
            if len(recent) > self.config.eval_every:
                recent.pop(0)

            if episode % self.config.eval_every == 0 or episode == self.config.episodes:
                metrics = self._metrics(episode, recent)
                self.logger.log(metrics)

        run_dir = Path(self.config.run_dir)
        checkpoint = run_dir / "policy.json"
        self.policy.save(checkpoint)
        train_report = Evaluator(self.env, self.policy, seed=self.config.seed + 1).run(
            self.config.eval_tasks,
            split="train",
        )
        eval_report = Evaluator(self.env, self.policy, seed=self.config.seed + 2).run(
            self.config.eval_tasks,
            split="eval",
        )
        summary = {
            "config": asdict(self.config),
            "checkpoint": str(checkpoint),
            "train_accuracy": train_report.accuracy,
            "eval_accuracy": eval_report.accuracy,
            "train_mean_reward": train_report.mean_reward,
            "eval_mean_reward": eval_report.mean_reward,
            "reward_hacking_findings": eval_report.reward_hacking_findings,
        }
        self.logger.write_summary(summary)
        return summary

    def _metrics(self, episode: int, recent: List[Transition]) -> Dict[str, Any]:
        rewards = [item.reward for item in recent]
        passes = [1.0 if item.passed else 0.0 for item in recent]
        train_report = Evaluator(self.env, self.policy, seed=self.config.seed + episode).run(
            self.config.eval_tasks,
            split="train",
        )
        eval_report = Evaluator(self.env, self.policy, seed=self.config.seed + episode + 1).run(
            self.config.eval_tasks,
            split="eval",
        )
        return {
            "episode": episode,
            "reward_mean": mean(rewards) if rewards else 0.0,
            "sample_accuracy": mean(passes) if passes else 0.0,
            "train_accuracy": train_report.accuracy,
            "eval_accuracy": eval_report.accuracy,
            "train_failures": train_report.failures_by_reason,
            "eval_failures": eval_report.failures_by_reason,
        }
