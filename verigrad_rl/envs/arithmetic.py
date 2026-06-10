"""Arithmetic text environment with train/eval split controls."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Sequence, Tuple

from verigrad_rl.envs.base import Task, TextEnv
from verigrad_rl.rewards import IntegerAnswerVerifier, Verification


@dataclass(frozen=True)
class ArithmeticSpec:
    max_number: int = 9
    eval_max_number: int = 14
    operations: Tuple[str, ...] = ("+", "-")


class ArithmeticEnv(TextEnv):
    """Prompt an agent to solve small arithmetic tasks as text."""

    name = "arithmetic"

    def __init__(self, spec: ArithmeticSpec | None = None) -> None:
        self.spec = spec or ArithmeticSpec()
        self.verifier = IntegerAnswerVerifier()

    def sample_task(self, rng: Random, split: str = "train") -> Task:
        if split == "eval":
            low = self.spec.max_number + 1
            high = self.spec.eval_max_number
        else:
            low = 0
            high = self.spec.max_number

        left = rng.randint(low, high)
        right = rng.randint(low, high)
        operation = rng.choice(list(self.spec.operations))
        answer = self._solve(left, operation, right)
        prompt = f"Solve exactly: {left} {operation} {right}"
        return Task(
            prompt=prompt,
            answer=str(answer),
            split=split,
            metadata={"left": left, "right": right, "operation": operation},
        )

    def candidate_actions(self) -> List[str]:
        lo, hi = self._answer_bounds(self.spec.eval_max_number, self.spec.operations)
        return [str(value) for value in range(lo, hi + 1)]

    def verify(self, task: Task, action: str) -> Verification:
        return self.verifier.verify(task.answer, action, task.metadata)

    @staticmethod
    def _solve(left: int, operation: str, right: int) -> int:
        if operation == "+":
            return left + right
        if operation == "-":
            return left - right
        if operation == "*":
            return left * right
        raise ValueError(f"Unsupported operation: {operation}")

    @staticmethod
    def _answer_bounds(max_number: int, operations: Sequence[str]) -> Tuple[int, int]:
        lows = []
        highs = []
        if "+" in operations:
            lows.append(0)
            highs.append(2 * max_number)
        if "-" in operations:
            lows.append(-max_number)
            highs.append(max_number)
        if "*" in operations:
            lows.append(0)
            highs.append(max_number * max_number)
        return min(lows), max(highs)
