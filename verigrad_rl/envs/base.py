"""Base contracts for text-agent RL environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List

from verigrad_rl.rewards import Verification


@dataclass(frozen=True)
class Task:
    """A single text-agent task with hidden verifier metadata."""

    prompt: str
    answer: str
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextEnv(ABC):
    """Finite-action text environment.

    A real LM backend can replace the finite action list with sampled completions.
    The interface is deliberately small so rewards, rollouts, and evals can stay
    independent from the model implementation.
    """

    name: str

    @abstractmethod
    def sample_task(self, rng: Random, split: str = "train") -> Task:
        """Sample one task from a named split."""

    @abstractmethod
    def candidate_actions(self) -> List[str]:
        """Return legal text actions for the baseline policy."""

    @abstractmethod
    def verify(self, task: Task, action: str) -> Verification:
        """Score an action for a task."""
