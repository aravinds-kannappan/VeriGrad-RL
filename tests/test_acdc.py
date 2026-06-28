"""Tests for the automated circuit-discovery system.

Two things must hold: path patching is exact (so the causal claims are real), and
ACDC recovers the known ground-truth circuit on the transparent safety DAG — the
white-box analogue of how Conmy et al. (2023) validate ACDC against hand-found
circuits in GPT-2 Small.
"""

import unittest

from verigrad_rl.mech.acdc import contrastive_dataset, run_acdc
from verigrad_rl.mech.circuit_graph import (
    SAFETY_DAG_DISTRACTORS,
    SAFETY_DAG_GROUND_TRUTH,
    from_toy_safety_circuit,
    safety_dag,
)

CLEAN = {"harm": 0.9, "jailbreak": 0.8, "topic": 0.1, "refusal_cue": 0.3, "noise": 0.5}
CORRUPT = {"harm": 0.1, "jailbreak": 0.1, "topic": 0.9, "refusal_cue": 0.3, "noise": 0.5}


class PathPatchingInvariantTests(unittest.TestCase):
    def setUp(self):
        self.g = safety_dag()

    def test_no_ablation_equals_clean(self):
        clean_acts = self.g.run(CLEAN)
        patched = self.g.run_patched(CLEAN, CORRUPT, ablated=set())
        for name in self.g.outputs:
            self.assertAlmostEqual(clean_acts[name], patched[name], places=9)

    def test_all_ablation_equals_corrupt(self):
        corrupt_acts = self.g.run(CORRUPT)
        patched = self.g.run_patched(CLEAN, CORRUPT, ablated=set(self.g.edges()))
        for name in self.g.outputs:
            self.assertAlmostEqual(corrupt_acts[name], patched[name], places=9)

    def test_patching_a_real_edge_moves_the_metric(self):
        clean_acts = self.g.run(CLEAN)
        base = self.g.kl_from_clean(clean_acts, self.g.run_patched(CLEAN, CORRUPT, set()))
        moved = self.g.kl_from_clean(
            clean_acts, self.g.run_patched(CLEAN, CORRUPT, {("threat", "guard")})
        )
        self.assertAlmostEqual(base, 0.0, places=9)
        self.assertGreater(moved, 0.05)  # threat->guard is on the behavior path


class ACDCRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.g = safety_dag()
        self.data = contrastive_dataset(self.g, CLEAN, CORRUPT, n=24, seed=1)

    def test_recovers_core_pathway(self):
        result = run_acdc(self.g, self.data, tau=0.02)
        found = set(result.circuit_edges)
        # The harm-detection -> guard -> output pathway must survive.
        for edge in [("harm", "threat"), ("threat", "guard"), ("guard", "refuse_logit")]:
            self.assertIn(edge, found, f"{edge} should be in the discovered circuit")

    def test_prunes_constant_input_edges(self):
        result = run_acdc(self.g, self.data, tau=0.02)
        found = set(result.circuit_edges)
        # refusal_cue and noise are identical across the contrast -> no information.
        for edge in SAFETY_DAG_DISTRACTORS:
            self.assertNotIn(edge, found, f"{edge} carries no info and should be pruned")

    def test_circuit_is_sparse_and_faithful(self):
        result = run_acdc(self.g, self.data, tau=0.02)
        self.assertLess(result.n_edges, result.full_edges)  # actually pruned something
        self.assertLess(result.faithfulness_kl, 0.02)       # still reproduces behavior
        scores = result.recovered(SAFETY_DAG_GROUND_TRUTH)
        # No spurious edges (every kept edge is genuinely behavior-relevant) and most
        # of the answer key recovered. Recall vs tau is the paper's precision/recall knob.
        self.assertGreaterEqual(scores["precision"], 0.85)
        self.assertGreaterEqual(scores["recall"], 0.7)

    def test_higher_tau_is_at_least_as_sparse(self):
        loose = run_acdc(self.g, self.data, tau=0.001)
        tight = run_acdc(self.g, self.data, tau=0.05)
        self.assertLessEqual(tight.n_edges, loose.n_edges)


class ToyCircuitGraphTests(unittest.TestCase):
    def test_lifts_and_runs(self):
        g = from_toy_safety_circuit()
        self.assertIn("harmful_completion", g.outputs)
        harmful = {"harmful_intent": 0.9, "helpful_intent": 0.3, "jailbreak_pressure": 0.9,
                   "refusal_prior": 0.2, "uncertainty": 0.2}
        benign = {"harmful_intent": 0.1, "helpful_intent": 0.9, "jailbreak_pressure": 0.1,
                  "refusal_prior": 0.2, "uncertainty": 0.2}
        data = contrastive_dataset(g, harmful, benign, n=16, seed=2)
        result = run_acdc(g, data, tau=0.02)
        found = set(result.circuit_edges)
        # harmful_intent must drive the harmful-completion logit on this contrast.
        self.assertIn(("harmful_intent", "harmful_completion"), found)


if __name__ == "__main__":
    unittest.main()
