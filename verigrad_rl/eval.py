"""Evaluation harnesses for trained policies."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Dict, List

from verigrad_rl.envs.base import TextEnv
from verigrad_rl.monitors import RewardHackingMonitor
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.rollout import Transition, collect_rollouts


@dataclass(frozen=True)
class EvalReport:
    env_name: str
    split: str
    tasks: int
    accuracy: float
    mean_reward: float
    failures_by_reason: Dict[str, int]
    reward_hacking_findings: int


class Evaluator:
    """Run deterministic policy evals and verifier probes."""

    def __init__(self, env: TextEnv, policy: SoftmaxTextPolicy, seed: int = 0) -> None:
        self.env = env
        self.policy = policy
        self.seed = seed

    def run(self, tasks: int = 100, split: str = "train") -> EvalReport:
        rng = Random(self.seed)
        transitions = collect_rollouts(self.env, self.policy, rng, tasks, split=split, greedy=True)
        failures = self._failures_by_reason(transitions)
        probe_rng = Random(self.seed + 10_000)
        findings = RewardHackingMonitor(self.env).sample_and_probe(probe_rng, count=min(tasks, 25))
        return EvalReport(
            env_name=self.env.name,
            split=split,
            tasks=tasks,
            accuracy=sum(1 for item in transitions if item.passed) / max(tasks, 1),
            mean_reward=mean([item.reward for item in transitions]) if transitions else 0.0,
            failures_by_reason=failures,
            reward_hacking_findings=len(findings),
        )

    @staticmethod
    def _failures_by_reason(transitions: List[Transition]) -> Dict[str, int]:
        failures: Dict[str, int] = {}
        for transition in transitions:
            if not transition.passed:
                failures[transition.reason] = failures.get(transition.reason, 0) + 1
        return failures
