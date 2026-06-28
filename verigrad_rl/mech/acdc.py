"""ACDC — Automated Circuit Discovery.

A faithful, dependency-free implementation of the core algorithm from Conmy,
Mavor-Parker, Lynch, Heimersheim & Garriga-Alonso, "Towards Automated Circuit
Discovery for Mechanistic Interpretability" (NeurIPS 2023, arXiv:2304.14997;
code: github.com/ArthurConmy/Automatic-Circuit-Discovery), run on the transparent
:class:`~verigrad_rl.mech.circuit_graph.CircuitGraph` substrate.

The idea, unchanged from the paper:

1. Pick a *task* — a dataset of (clean, corrupt) input pairs that contrasts the
   behavior you want to explain — and a *metric* (here, KL of the output softmax
   from the clean run).
2. Walk the computational graph in **reverse topological order**. For each incoming
   edge of each node, try removing it by **path-patching** that edge to the corrupt
   activation. If the marginal increase in the metric is below a threshold ``tau``,
   the edge is unimportant: prune it.
3. The edges that survive are the discovered circuit.

Because the substrate is white-box with a known answer key (``SAFETY_DAG_GROUND_TRUTH``),
the discovery can be *validated* the way the paper validates ACDC against
hand-identified circuits — recovering the behavior-relevant edges while pruning the
ones carrying no information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from verigrad_rl.mech.circuit_graph import CircuitGraph, Edge

# One task example: a clean input and the corrupt counterfactual it is contrasted with.
Example = Tuple[Dict[str, float], Dict[str, float]]


@dataclass
class ACDCResult:
    """What a discovery run found."""

    circuit_edges: List[Edge]
    pruned_edges: List[Edge]
    faithfulness_kl: float  # KL(clean || circuit-only) — lower is more faithful
    full_edges: int
    tau: float
    metrics_trace: List[float] = field(default_factory=list)

    @property
    def n_edges(self) -> int:
        return len(self.circuit_edges)

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_edges / max(self.full_edges, 1)

    def recovered(self, ground_truth: Sequence[Edge]) -> Dict[str, float]:
        """Precision/recall of the discovered circuit against an answer key."""

        found = set(self.circuit_edges)
        truth = set(ground_truth)
        tp = len(found & truth)
        precision = tp / max(len(found), 1)
        recall = tp / max(len(truth), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"precision": precision, "recall": recall, "f1": f1}


def _mean_kl(graph: CircuitGraph, dataset: Sequence[Example], ablated: set[Edge]) -> float:
    """Average over the task of KL(clean output || patched output) for an ablation set."""

    total = 0.0
    for clean, corrupt in dataset:
        clean_acts = graph.run(clean)
        patched_acts = graph.run_patched(clean, corrupt, ablated)
        total += graph.kl_from_clean(clean_acts, patched_acts)
    return total / max(len(dataset), 1)


def run_acdc(
    graph: CircuitGraph,
    dataset: Sequence[Example],
    tau: float = 0.02,
) -> ACDCResult:
    """Greedy reverse-topological edge pruning (the ACDC algorithm).

    ``tau`` is the KL budget per edge, in nats: an edge whose removal raises the
    metric by less than ``tau`` over the running circuit is pruned. Larger ``tau``
    -> sparser, more aggressive circuits (the paper's central recall/precision knob).
    """

    all_edges = graph.edges()
    ablated: set[Edge] = set()
    trace: List[float] = []

    # Reverse topological order over edges: visit children latest-first, and for a
    # given child consider its incoming edges. graph.edges() is in child-topo order,
    # so reversing gives reverse-topological traversal.
    for edge in reversed(all_edges):
        base = _mean_kl(graph, dataset, ablated)
        trial = ablated | {edge}
        score = _mean_kl(graph, dataset, trial)
        if score - base < tau:
            ablated.add(edge)
        trace.append(score)

    circuit_edges = [e for e in all_edges if e not in ablated]
    faithfulness = _mean_kl(graph, dataset, ablated)
    return ACDCResult(
        circuit_edges=circuit_edges,
        pruned_edges=sorted(ablated),
        faithfulness_kl=faithfulness,
        full_edges=len(all_edges),
        tau=tau,
        metrics_trace=trace,
    )


def contrastive_dataset(
    graph: CircuitGraph,
    clean_template: Dict[str, float],
    corrupt_template: Dict[str, float],
    n: int = 16,
    jitter: float = 0.05,
    seed: int = 0,
) -> List[Example]:
    """Build a task: ``n`` jittered (clean, corrupt) pairs around two templates.

    Inputs present in both templates with the *same* value are held constant across
    the contrast (their edges should be pruned); inputs that differ drive the
    behavior the circuit explains.
    """

    from random import Random

    rng = Random(seed)
    keys = graph.inputs
    examples: List[Example] = []
    for _ in range(n):
        clean = {}
        corrupt = {}
        for k in keys:
            cv = clean_template.get(k, 0.0)
            dv = corrupt_template.get(k, 0.0)
            if abs(cv - dv) < 1e-9:
                # held constant: identical sample in both runs
                shared = cv + rng.uniform(-jitter, jitter)
                clean[k] = shared
                corrupt[k] = shared
            else:
                clean[k] = cv + rng.uniform(-jitter, jitter)
                corrupt[k] = dv + rng.uniform(-jitter, jitter)
        examples.append((clean, corrupt))
    return examples
