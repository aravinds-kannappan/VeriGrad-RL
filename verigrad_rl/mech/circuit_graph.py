"""A transparent computational graph + path patching.

This is the substrate the automated circuit-discovery system runs on. A
``CircuitGraph`` is an explicit DAG of named nodes; every node knows its parents
and the function that turns parent activations into its own. Because the graph is
fully white-box, *path patching*, the causal primitive from Goldowsky-Dill et al.,
"Localizing Model Behavior with Path Patching" (Redwood Research, arXiv:2304.05969,
2023), has an exact, testable implementation here rather than an approximation.

Edge patching (the form ACDC uses): the activation flowing along edge ``(u -> v)``
is either ``u``'s value on the *clean* input (edge kept) or on the *corrupt* input
(edge ablated). Kept edges carry the recovered upstream activation, so effects
propagate through recomputation. Two invariants follow and are unit-tested:

- ablate **no** edges  -> the patched run equals the clean run;
- ablate **all** edges -> the patched run equals the corrupt run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

Activations = Dict[str, float]
Edge = Tuple[str, str]  # (parent, child)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


@dataclass(frozen=True)
class Node:
    """One unit of computation in the circuit graph.

    ``fn`` maps a dict of parent activations to this node's scalar activation.
    Input nodes have no parents and ``fn`` is ignored (their value is the input).
    """

    name: str
    parents: Tuple[str, ...]
    fn: Callable[[Activations], float] | None = None
    layer: int = 0

    @property
    def is_input(self) -> bool:
        return len(self.parents) == 0


class CircuitGraph:
    """A small feed-forward DAG with exact path patching."""

    def __init__(self, nodes: Sequence[Node], outputs: Sequence[str]) -> None:
        self.nodes: Dict[str, Node] = {n.name: n for n in nodes}
        self.outputs: Tuple[str, ...] = tuple(outputs)
        self.inputs: Tuple[str, ...] = tuple(n.name for n in nodes if n.is_input)
        self._order: List[str] = self._topo_sort()

    # -- structure ---------------------------------------------------------- #
    def edges(self) -> List[Edge]:
        """Every internal (patchable) edge, in topological child order."""

        out: List[Edge] = []
        for name in self._order:
            for parent in self.nodes[name].parents:
                out.append((parent, name))
        return out

    def _topo_sort(self) -> List[str]:
        seen: List[str] = []
        temp: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            if name in temp:
                raise ValueError(f"cycle through {name!r}; graph must be a DAG")
            temp.add(name)
            for parent in self.nodes[name].parents:
                visit(parent)
            temp.discard(name)
            seen.append(name)

        for name in self.nodes:
            visit(name)
        return seen

    # -- forward passes ----------------------------------------------------- #
    def run(self, inputs: Activations) -> Activations:
        """Clean forward pass: activations at every node for ``inputs``."""

        acts: Activations = {}
        for name in self._order:
            node = self.nodes[name]
            if node.is_input:
                acts[name] = float(inputs[name])
            else:
                assert node.fn is not None
                acts[name] = node.fn({p: acts[p] for p in node.parents})
        return acts

    def run_patched(
        self, clean: Activations, corrupt: Activations, ablated: set[Edge]
    ) -> Activations:
        """Edge-level path patching.

        For every node, each parent edge contributes the parent's *corrupt*
        activation if the edge is in ``ablated``, else the recovered patched
        (clean-lineage) activation. See module docstring for the invariants.
        """

        corrupt_acts = self.run(corrupt)
        patched: Activations = {}
        for name in self._order:
            node = self.nodes[name]
            if node.is_input:
                patched[name] = float(clean[name])
                continue
            assert node.fn is not None
            parent_vals = {
                p: (corrupt_acts[p] if (p, name) in ablated else patched[p])
                for p in node.parents
            }
            patched[name] = node.fn(parent_vals)
        return patched

    # -- metric ------------------------------------------------------------- #
    def output_logits(self, acts: Activations) -> List[float]:
        return [acts[o] for o in self.outputs]

    def kl_from_clean(self, clean_acts: Activations, other_acts: Activations) -> float:
        """KL(softmax(clean outputs) || softmax(other outputs)): the ACDC metric."""

        return kl_divergence(self.output_logits(clean_acts), self.output_logits(other_acts))


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def kl_divergence(p_logits: Sequence[float], q_logits: Sequence[float]) -> float:
    """KL(P || Q) where P, Q are softmax of the given logits (in nats)."""

    p = softmax(p_logits)
    q = softmax(q_logits)
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            total += pi * math.log(pi / max(qi, 1e-12))
    return total


# --------------------------------------------------------------------------- #
# Reference circuits with known ground truth                                  #
# --------------------------------------------------------------------------- #
def safety_dag() -> CircuitGraph:
    """A 4-layer refuse/answer safety circuit with a known ground-truth circuit.

    Inputs:  harm, jailbreak, topic (benign-content signal), refusal_cue, noise.
    Hidden:  threat = relu(1.4 harm + 1.0 jailbreak - 0.5 topic)
             benign = relu(1.3 topic - 0.8 harm)
             guard  = relu(1.1 threat + 0.6 refusal_cue - 0.4 benign)
    Outputs: refuse_logit = 1.5 guard - 0.5 benign + 0.01 noise
             answer_logit = 1.4 benign - 1.2 guard

    On a harmful-vs-benign contrast (harm/topic differ, refusal_cue + noise held
    constant) the behavior-relevant circuit is the harm-detection -> guard -> output
    pathway; edges out of the constant inputs carry no information and ACDC prunes
    them. That gives this graph a checkable answer key.
    """

    nodes = [
        Node("harm", ()),
        Node("jailbreak", ()),
        Node("topic", ()),
        Node("refusal_cue", ()),
        Node("noise", ()),
        Node(
            "threat",
            ("harm", "jailbreak", "topic"),
            lambda a: relu(1.4 * a["harm"] + 1.0 * a["jailbreak"] - 0.5 * a["topic"]),
            layer=1,
        ),
        Node(
            "benign",
            ("topic", "harm"),
            lambda a: relu(1.3 * a["topic"] - 0.8 * a["harm"]),
            layer=1,
        ),
        Node(
            "guard",
            ("threat", "refusal_cue", "benign"),
            lambda a: relu(1.1 * a["threat"] + 0.6 * a["refusal_cue"] - 0.4 * a["benign"]),
            layer=2,
        ),
        Node(
            "refuse_logit",
            ("guard", "benign", "noise"),
            lambda a: 1.5 * a["guard"] - 0.5 * a["benign"] + 0.01 * a["noise"],
            layer=3,
        ),
        Node(
            "answer_logit",
            ("benign", "guard"),
            lambda a: 1.4 * a["benign"] - 1.2 * a["guard"],
            layer=3,
        ),
    ]
    return CircuitGraph(nodes, outputs=("refuse_logit", "answer_logit"))


# The behaviour-relevant edges a correct discovery should keep on the harmful-vs-
# benign contrast (used to validate the discovery system, mirroring how ACDC is
# validated against hand-found circuits in Conmy et al., 2023).
SAFETY_DAG_GROUND_TRUTH: Tuple[Edge, ...] = (
    ("harm", "threat"),
    ("jailbreak", "threat"),
    ("topic", "threat"),
    ("topic", "benign"),
    ("harm", "benign"),
    ("threat", "guard"),
    ("benign", "guard"),
    ("guard", "refuse_logit"),
    ("benign", "refuse_logit"),
    ("benign", "answer_logit"),
    ("guard", "answer_logit"),
)

# Edges out of inputs held constant across the contrast; a correct run prunes them.
SAFETY_DAG_DISTRACTORS: Tuple[Edge, ...] = (
    ("refusal_cue", "guard"),
    ("noise", "refuse_logit"),
)


def from_toy_safety_circuit() -> CircuitGraph:
    """Lift the RL env's ``ToySafetyCircuit`` (features -> logits) into a graph.

    Single layer, but it lets ACDC run on the *same* circuit the safety-steering RL
    environment uses, so a discovered circuit is about the real reward model.
    """

    from verigrad_rl.mech.activations import ActivationSnapshot, ToySafetyCircuit

    circuit = ToySafetyCircuit()
    features = ("harmful_intent", "helpful_intent", "jailbreak_pressure", "refusal_prior", "uncertainty")
    # Read the linear weights straight off the circuit by probing one feature at a time.
    base = {f: 0.0 for f in features}
    base_logits = circuit.logits(ActivationSnapshot(base))
    weights: Dict[str, Dict[str, float]] = {o: {} for o in base_logits}

    for f in features:
        probe = dict(base)
        probe[f] = 1.0
        logits = circuit.logits(ActivationSnapshot(probe))
        for o, v in logits.items():
            weights[o][f] = v - base_logits[o]

    nodes: List[Node] = [Node(f, ()) for f in features]
    for o in base_logits:
        w = weights[o]

        def make_fn(weight_row: Dict[str, float]) -> Callable[[Activations], float]:
            return lambda a, _w=weight_row: sum(_w[p] * a[p] for p in _w if abs(_w[p]) > 1e-9)

        parents = tuple(f for f in features if abs(w[f]) > 1e-9)
        nodes.append(Node(o, parents, make_fn(w), layer=1))
    return CircuitGraph(nodes, outputs=tuple(base_logits.keys()))
