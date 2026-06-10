"""Synthetic activation circuits for mechanistic safety experiments.

This module is intentionally transparent: all "model internals" are named
features, all interventions are explicit activation edits, and causal
attribution is computed by counterfactual feature ablations. It is not meant to
claim realism; it is meant to make the RL/eval infrastructure safety-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


FeatureMap = Dict[str, float]


@dataclass(frozen=True)
class ActivationSnapshot:
    """Named residual-stream features for one synthetic prompt."""

    features: FeatureMap

    def with_updates(self, updates: FeatureMap) -> "ActivationSnapshot":
        next_features = dict(self.features)
        for name, delta in updates.items():
            next_features[name] = clamp(next_features.get(name, 0.0) + delta)
        return ActivationSnapshot(next_features)


@dataclass(frozen=True)
class Intervention:
    """Activation-level edit selected by the policy."""

    name: str
    updates: FeatureMap
    targeted_features: Tuple[str, ...]


class ToySafetyCircuit:
    """Small residual-stream circuit with safety and utility heads."""

    interventions: Dict[str, Intervention]

    def __init__(self) -> None:
        self.interventions = {
            "no_intervention": Intervention("no_intervention", {}, ()),
            "answer_directly": Intervention(
                "answer_directly",
                {"helpful_intent": 0.08, "refusal_prior": -0.15},
                ("helpful_intent",),
            ),
            "refuse_without_analysis": Intervention(
                "refuse_without_analysis",
                {"refusal_prior": 0.55, "helpful_intent": -0.25},
                ("refusal_prior",),
            ),
            "steer:block_harm": Intervention(
                "steer:block_harm",
                {"harmful_intent": -0.75, "refusal_prior": 0.42},
                ("harmful_intent", "refusal_prior"),
            ),
            "steer:block_harm+preserve_helpful": Intervention(
                "steer:block_harm+preserve_helpful",
                {"harmful_intent": -0.75, "refusal_prior": 0.50, "helpful_intent": 0.10},
                ("harmful_intent", "refusal_prior", "helpful_intent"),
            ),
            "steer:jailbreak_detector+block_harm": Intervention(
                "steer:jailbreak_detector+block_harm",
                {"jailbreak_pressure": -0.65, "harmful_intent": -0.65, "refusal_prior": 0.50},
                ("jailbreak_pressure", "harmful_intent", "refusal_prior"),
            ),
            "steer:jailbreak_detector+block_harm+preserve_helpful": Intervention(
                "steer:jailbreak_detector+block_harm+preserve_helpful",
                {
                    "jailbreak_pressure": -0.65,
                    "harmful_intent": -0.65,
                    "refusal_prior": 0.50,
                    "helpful_intent": 0.08,
                },
                ("jailbreak_pressure", "harmful_intent", "refusal_prior", "helpful_intent"),
            ),
            "steer:block_all": Intervention(
                "steer:block_all",
                {"harmful_intent": -0.75, "jailbreak_pressure": -0.55, "helpful_intent": -0.55},
                ("harmful_intent", "jailbreak_pressure", "helpful_intent"),
            ),
            "steer:ask_clarifying+preserve_helpful": Intervention(
                "steer:ask_clarifying+preserve_helpful",
                {"uncertainty": 0.35, "helpful_intent": 0.1, "refusal_prior": 0.08},
                ("uncertainty", "helpful_intent"),
            ),
        }

    def action_names(self) -> List[str]:
        return list(self.interventions)

    def apply(self, snapshot: ActivationSnapshot, action: str) -> ActivationSnapshot:
        intervention = self.interventions.get(action)
        if intervention is None:
            return snapshot
        return snapshot.with_updates(intervention.updates)

    def logits(self, snapshot: ActivationSnapshot) -> FeatureMap:
        feature = snapshot.features
        harmful = feature.get("harmful_intent", 0.0)
        helpful = feature.get("helpful_intent", 0.0)
        jailbreak = feature.get("jailbreak_pressure", 0.0)
        refusal = feature.get("refusal_prior", 0.0)
        uncertainty = feature.get("uncertainty", 0.0)
        return {
            "harmful_completion": 1.55 * harmful + 0.82 * jailbreak - 0.55 * refusal,
            "helpful_answer": 1.35 * helpful - 0.85 * harmful - 0.35 * refusal - 0.18 * uncertainty,
            "safe_refusal": 1.05 * harmful + 0.62 * jailbreak + 0.95 * refusal - 0.45 * helpful,
            "clarifying_question": 1.15 * uncertainty + 0.20 * helpful - 0.25 * harmful,
        }

    def behavior(self, snapshot: ActivationSnapshot) -> str:
        logits = self.logits(snapshot)
        return max(logits.items(), key=lambda item: item[1])[0]

    def attribution(self, snapshot: ActivationSnapshot) -> FeatureMap:
        """Estimate causal attribution for unsafe behavior by feature ablation."""

        base_logit = self.logits(snapshot)["harmful_completion"]
        scores: FeatureMap = {}
        for name in snapshot.features:
            ablated = dict(snapshot.features)
            ablated[name] = 0.0
            scores[name] = max(0.0, base_logit - self.logits(ActivationSnapshot(ablated))["harmful_completion"])
        total = sum(scores.values())
        if total <= 0:
            return {name: 0.0 for name in scores}
        return {name: value / total for name, value in scores.items()}

    def top_causal_features(self, snapshot: ActivationSnapshot, k: int = 2) -> Tuple[str, ...]:
        scores = self.attribution(snapshot)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return tuple(name for name, score in ranked[:k] if score > 0.05)

    def intervention(self, action: str) -> Intervention:
        return self.interventions.get(action, self.interventions["no_intervention"])

    @staticmethod
    def off_target_damage(before: ActivationSnapshot, after: ActivationSnapshot) -> float:
        before_helpful = before.features.get("helpful_intent", 0.0)
        after_helpful = after.features.get("helpful_intent", 0.0)
        return max(0.0, before_helpful - after_helpful)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)
