"""Synthetic biosafety triage circuit.

The model here is deliberately non-operational. It uses mock risk features and
synthetic sequence-similarity signals to demonstrate how VeriGrad RL can be
applied to real-world safety workflows without encoding actionable biology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from verigrad_rl.mech.activations import ActivationSnapshot, FeatureMap, Intervention, clamp


@dataclass(frozen=True)
class BioSafetyDecision:
    behavior: str
    logits: FeatureMap


class ToyBioSafetyCircuit:
    """Toy circuit for DNA-order and dual-use request triage."""

    interventions: Dict[str, Intervention]

    def __init__(self) -> None:
        self.interventions = {
            "approve_low_risk": Intervention(
                "approve_low_risk",
                {"benign_research_value": 0.12, "review_pressure": -0.18},
                ("benign_research_value",),
            ),
            "request_more_context": Intervention(
                "request_more_context",
                {"uncertainty": 0.24, "review_pressure": 0.18},
                ("uncertainty", "review_pressure"),
            ),
            "route_biosafety_review": Intervention(
                "route_biosafety_review",
                {"review_pressure": 0.48, "sequence_risk": -0.08, "synthesis_scale": -0.04},
                ("review_pressure", "sequence_risk", "synthesis_scale"),
            ),
            "block_and_explain_safely": Intervention(
                "block_and_explain_safely",
                {"review_pressure": 0.62, "sequence_risk": -0.20, "capability_risk": -0.18},
                ("review_pressure", "sequence_risk", "capability_risk"),
            ),
            "redact_and_offer_safe_alternative": Intervention(
                "redact_and_offer_safe_alternative",
                {"capability_risk": -0.35, "benign_research_value": 0.18, "review_pressure": 0.24},
                ("capability_risk", "benign_research_value", "review_pressure"),
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
        sequence = feature.get("sequence_risk", 0.0)
        capability = feature.get("capability_risk", 0.0)
        scale = feature.get("synthesis_scale", 0.0)
        value = feature.get("benign_research_value", 0.0)
        uncertainty = feature.get("uncertainty", 0.0)
        verification = feature.get("customer_verification", 0.0)
        documentation = feature.get("documentation_quality", 0.0)
        review = feature.get("review_pressure", 0.0)
        return {
            "approve_order": 1.65 * value + 0.44 * verification + 0.28 * documentation - 1.15 * sequence - 0.80 * capability - 0.45 * scale - 0.55 * review,
            "request_context": 1.60 * uncertainty + 0.65 * (1.0 - documentation) + 0.32 * value + 0.20 * review - 0.25 * sequence,
            "route_review": 1.10 * sequence + 0.72 * capability + 0.60 * scale + 0.42 * (1.0 - verification) + 1.25 * review,
            "block_request": 1.30 * sequence + 1.10 * capability + 0.52 * scale + 0.32 * (1.0 - verification) + 0.20 * review - 0.35 * value,
            "safe_education": 0.75 * value + 0.52 * capability + 0.22 * uncertainty + 0.14 * documentation - 0.30 * sequence,
        }

    def behavior(self, snapshot: ActivationSnapshot) -> str:
        logits = self.logits(snapshot)
        return max(logits.items(), key=lambda item: item[1])[0]

    def attribution(self, snapshot: ActivationSnapshot) -> FeatureMap:
        """Estimate risk attribution for block/review behavior."""

        logits = self.logits(snapshot)
        base = max(logits["block_request"], logits["route_review"])
        scores: FeatureMap = {}
        for name in snapshot.features:
            ablated = dict(snapshot.features)
            ablated[name] = 0.0
            ablated_logits = self.logits(ActivationSnapshot(ablated))
            ablated_risk = max(ablated_logits["block_request"], ablated_logits["route_review"])
            scores[name] = max(0.0, base - ablated_risk)
        total = sum(scores.values())
        if total <= 0:
            return {name: 0.0 for name in scores}
        return {name: value / total for name, value in scores.items()}

    def top_causal_features(self, snapshot: ActivationSnapshot, k: int = 3) -> Tuple[str, ...]:
        ranked = sorted(self.attribution(snapshot).items(), key=lambda item: item[1], reverse=True)
        return tuple(name for name, score in ranked[:k] if score > 0.05)

    def intervention(self, action: str) -> Intervention:
        return self.interventions.get(action, self.interventions["request_more_context"])


def snapshot_from_slider_values(
    sequence_risk: float,
    capability_risk: float,
    synthesis_scale: float,
    benign_research_value: float,
    uncertainty: float,
    customer_verification: float = 0.5,
    documentation_quality: float = 0.5,
    review_pressure: float = 0.0,
) -> ActivationSnapshot:
    return ActivationSnapshot(
        {
            "sequence_risk": clamp(sequence_risk),
            "capability_risk": clamp(capability_risk),
            "synthesis_scale": clamp(synthesis_scale),
            "benign_research_value": clamp(benign_research_value),
            "uncertainty": clamp(uncertainty),
            "customer_verification": clamp(customer_verification),
            "documentation_quality": clamp(documentation_quality),
            "review_pressure": clamp(review_pressure),
        }
    )
