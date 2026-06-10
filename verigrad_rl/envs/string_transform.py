"""String transformation environment for verifier and eval experiments."""

from __future__ import annotations

from random import Random
from typing import List, Tuple

from verigrad_rl.envs.base import Task, TextEnv
from verigrad_rl.rewards import ExactAnswerVerifier, Verification


class StringTransformEnv(TextEnv):
    """Ask the agent to apply simple deterministic string transformations."""

    name = "string_transform"

    def __init__(self) -> None:
        self.words = ["orbit", "vector", "policy", "reward", "agent", "signal"]
        self.transforms: Tuple[str, ...] = ("reverse", "uppercase")
        self.verifier = ExactAnswerVerifier()

    def sample_task(self, rng: Random, split: str = "train") -> Task:
        word = rng.choice(self.words)
        transform = rng.choice(self.transforms)
        answer = word[::-1] if transform == "reverse" else word.upper()
        prompt = f"{transform} the token: {word}"
        return Task(prompt=prompt, answer=answer, split=split, metadata={"transform": transform})

    def candidate_actions(self) -> List[str]:
        actions = set()
        for word in self.words:
            actions.add(word[::-1])
            actions.add(word.upper())
        return sorted(actions)

    def verify(self, task: Task, action: str) -> Verification:
        return self.verifier.verify(task.answer, action, task.metadata)
