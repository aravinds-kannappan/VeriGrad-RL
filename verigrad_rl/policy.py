"""Small text policy used by the dependency-free trainer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Tuple


Feature = str


def extract_prompt_features(prompt: str) -> List[Feature]:
    """Extract sparse features from a text prompt."""

    parsed = parse_arithmetic(prompt)
    normalized = prompt.lower().strip()
    tokens = re.findall(r"[a-z]+|[-+*]|\d+", normalized)
    features = ["prompt:bias"]
    features.extend(f"prompt:tok={token}" for token in tokens)

    if parsed is not None:
        left, op, right, expected = parsed
        features.extend(
            [
                f"prompt:left={left}",
                f"prompt:right={right}",
                f"prompt:op={op}",
                f"prompt:triple={left}:{op}:{right}",
                f"prompt:computed={expected}",
            ]
        )
    return features


def extract_features(prompt: str, action: str) -> List[Feature]:
    """Extract sparse action-conditioned features.

    A larger LM backend would provide its own representation. For the default
    baseline, action-conditioned features let RL learn reusable concepts such as
    "candidate equals the parsed arithmetic result" instead of memorizing every
    prompt-action pair separately.
    """

    features = extract_prompt_features(prompt)
    features.append(f"action:text={action}")

    parsed = parse_arithmetic(prompt)
    if parsed is None:
        return features

    _, _, _, expected = parsed
    try:
        candidate = int(action.strip())
    except ValueError:
        features.append("action:non_integer")
        return features

    error = candidate - expected
    abs_error = abs(error)
    features.extend(
        [
            f"action:value={candidate}",
            f"arith:error={error}",
            f"arith:abs_error={min(abs_error, 5)}",
            "arith:exact" if error == 0 else "arith:not_exact",
            "arith:too_high" if error > 0 else "arith:not_too_high",
            "arith:too_low" if error < 0 else "arith:not_too_low",
        ]
    )
    return features


def parse_arithmetic(prompt: str) -> Optional[Tuple[int, str, int, int]]:
    match = re.search(r"(-?\d+)\s*([+*-])\s*(-?\d+)", prompt)
    if not match:
        return None
    left = int(match.group(1))
    op = match.group(2)
    right = int(match.group(3))
    if op == "+":
        expected = left + right
    elif op == "-":
        expected = left - right
    elif op == "*":
        expected = left * right
    else:
        return None
    return left, op, right, expected


@dataclass
class PolicyDecision:
    action: str
    logprob: float
    probabilities: Dict[str, float]


class SoftmaxTextPolicy:
    """Feature-hashed categorical policy over text actions."""

    def __init__(self, actions: Iterable[str], temperature: float = 1.0) -> None:
        self.actions = list(actions)
        if not self.actions:
            raise ValueError("SoftmaxTextPolicy requires at least one action")
        self.action_to_index = {action: idx for idx, action in enumerate(self.actions)}
        self.temperature = temperature
        self.weights: Dict[Feature, float] = {}

    def decide(self, prompt: str, rng: Random, greedy: bool = False) -> PolicyDecision:
        probabilities = self.distribution(prompt)
        if greedy:
            action = max(probabilities.items(), key=lambda item: item[1])[0]
        else:
            action = self._sample(probabilities, rng)
        return PolicyDecision(action, math.log(max(probabilities[action], 1e-12)), probabilities)

    def distribution(self, prompt: str) -> Dict[str, float]:
        scores = [self._score(prompt, action) / self.temperature for action in self.actions]
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        normalizer = sum(exp_scores)
        return {
            action: exp_score / normalizer
            for action, exp_score in zip(self.actions, exp_scores)
        }

    def update(self, prompt: str, action: str, advantage: float, learning_rate: float) -> None:
        if action not in self.action_to_index:
            raise ValueError(f"Unknown action: {action}")
        probabilities = self.distribution(prompt)
        expected_features: Dict[Feature, float] = {}
        for candidate, probability in probabilities.items():
            for feature in extract_features(prompt, candidate):
                expected_features[feature] = expected_features.get(feature, 0.0) + probability

        for feature in extract_features(prompt, action):
            self.weights[feature] = self.weights.get(feature, 0.0) + learning_rate * advantage
        for feature, expected_value in expected_features.items():
            self.weights[feature] = (
                self.weights.get(feature, 0.0) - learning_rate * advantage * expected_value
            )

    def save(self, path: str | Path) -> None:
        payload = {
            "actions": self.actions,
            "temperature": self.temperature,
            "weights": self.weights,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxTextPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(payload["actions"], temperature=payload.get("temperature", 1.0))
        policy.weights = {str(key): float(value) for key, value in payload["weights"].items()}
        return policy

    def _score(self, prompt: str, action: str) -> float:
        return sum(self.weights.get(feature, 0.0) for feature in extract_features(prompt, action))

    @staticmethod
    def _sample(probabilities: Dict[str, float], rng: Random) -> str:
        threshold = rng.random()
        cumulative = 0.0
        last_action = ""
        for action, probability in probabilities.items():
            cumulative += probability
            last_action = action
            if cumulative >= threshold:
                return action
        return last_action

    def top_actions(self, prompt: str, k: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.distribution(prompt).items(), key=lambda item: item[1], reverse=True)[:k]
