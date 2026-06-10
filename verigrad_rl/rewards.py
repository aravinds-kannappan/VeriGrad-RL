"""Reward and verifier contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Protocol


@dataclass(frozen=True)
class Verification:
    """Structured result from a reward verifier."""

    reward: float
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    """Protocol for verifiers that can score text actions."""

    def verify(self, expected: str, action: str, metadata: Dict[str, Any]) -> Verification:
        """Return reward and pass/fail details."""


def normalize_text(value: str) -> str:
    """Normalize model text for exact-match verifiers."""

    return " ".join(value.strip().lower().split())


class ExactAnswerVerifier:
    """Strict exact-answer verifier with basic format diagnostics."""

    def __init__(self, correct_reward: float = 1.0, wrong_reward: float = 0.0) -> None:
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward

    def verify(self, expected: str, action: str, metadata: Dict[str, Any]) -> Verification:
        normalized_action = normalize_text(action)
        normalized_expected = normalize_text(expected)

        if normalized_action == "":
            return Verification(self.wrong_reward, False, "blank_output")

        if normalized_action == normalized_expected:
            return Verification(self.correct_reward, True, "exact_match")

        return Verification(
            self.wrong_reward,
            False,
            "wrong_answer",
            {"expected": normalized_expected, "got": normalized_action, **metadata},
        )


class IntegerAnswerVerifier(ExactAnswerVerifier):
    """Exact verifier that rejects non-integer answers before correctness scoring."""

    def __init__(
        self,
        correct_reward: float = 1.0,
        wrong_reward: float = 0.0,
        shaped_reward_cap: float = 0.25,
        shaping_scale: float = 10.0,
    ) -> None:
        super().__init__(correct_reward=correct_reward, wrong_reward=wrong_reward)
        self.shaped_reward_cap = shaped_reward_cap
        self.shaping_scale = shaping_scale

    def verify(self, expected: str, action: str, metadata: Dict[str, Any]) -> Verification:
        normalized_action = normalize_text(action)
        try:
            parsed_action = int(normalized_action)
        except ValueError:
            return Verification(0.0, False, "non_integer_output", {"got": normalized_action})
        exact = super().verify(expected, action, metadata)
        if exact.passed:
            return exact
        parsed_expected = int(normalize_text(expected))
        distance = abs(parsed_action - parsed_expected)
        shaped_reward = max(0.0, 1.0 - distance / self.shaping_scale) * self.shaped_reward_cap
        return Verification(
            shaped_reward,
            False,
            "wrong_integer",
            {
                "expected": parsed_expected,
                "got": parsed_action,
                "distance": distance,
                **metadata,
            },
        )


class CompositeVerifier:
    """Run several verifier checks and return the first failure."""

    def __init__(self, verifiers: Iterable[Verifier]) -> None:
        self.verifiers = list(verifiers)

    def verify(self, expected: str, action: str, metadata: Dict[str, Any]) -> Verification:
        last_result = Verification(0.0, False, "no_verifiers")
        for verifier in self.verifiers:
            result = verifier.verify(expected, action, metadata)
            last_result = result
            if not result.passed:
                return result
        return last_result
