"""Rollout collection for text-agent policies."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List

from verigrad_rl.envs.base import Task, TextEnv
from verigrad_rl.policy import SoftmaxTextPolicy


@dataclass(frozen=True)
class Transition:
    task: Task
    action: str
    reward: float
    passed: bool
    reason: str
    logprob: float
    metadata: Dict[str, object]


def collect_one(
    env: TextEnv,
    policy: SoftmaxTextPolicy,
    rng: Random,
    split: str = "train",
    greedy: bool = False,
) -> Transition:
    task = env.sample_task(rng, split=split)
    decision = policy.decide(task.prompt, rng, greedy=greedy)
    verification = env.verify(task, decision.action)
    return Transition(
        task=task,
        action=decision.action,
        reward=verification.reward,
        passed=verification.passed,
        reason=verification.reason,
        logprob=decision.logprob,
        metadata=verification.details,
    )


def collect_rollouts(
    env: TextEnv,
    policy: SoftmaxTextPolicy,
    rng: Random,
    count: int,
    split: str = "train",
    greedy: bool = False,
) -> List[Transition]:
    return [collect_one(env, policy, rng, split=split, greedy=greedy) for _ in range(count)]
