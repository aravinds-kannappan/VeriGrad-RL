"""Offline tests for the scalable harness (no network, no API)."""

import tempfile
import unittest
from pathlib import Path
from random import Random

from verigrad_rl.propensity import stats
from verigrad_rl.propensity.scale import core
from verigrad_rl.propensity.scale.core import Task, score
from verigrad_rl.propensity.scale.environments import CommonsenseQAEnvironment, GSM8KEnvironment
from verigrad_rl.propensity.scale.pressures import AuthorityWrongPressure, HonestPressure
from verigrad_rl.propensity.scale.store import Store, cache_key


class TestStats(unittest.TestCase):
    def test_benjamini_hochberg_basic(self):
        rejected, q = stats.benjamini_hochberg([0.001, 0.2, 0.7, 0.9], alpha=0.05)
        self.assertTrue(rejected[0])
        self.assertFalse(rejected[1])
        self.assertTrue(all(0.0 <= x <= 1.0 for x in q))

    def test_benjamini_hochberg_all_null(self):
        rejected, _ = stats.benjamini_hochberg([0.6, 0.7, 0.8], alpha=0.05)
        self.assertEqual(rejected, [False, False, False])

    def test_two_proportion(self):
        self.assertAlmostEqual(stats.two_proportion_p(10, 100, 10, 100), 1.0, places=6)
        self.assertLess(stats.two_proportion_p(5, 100, 60, 100), 0.001)

    def test_cluster_bootstrap(self):
        pt, (lo, hi) = stats.cluster_bootstrap_ci([[True, True], [True, False], [True, True]])
        self.assertAlmostEqual(pt, 5 / 6, places=6)
        self.assertLessEqual(lo, pt)
        self.assertGreaterEqual(hi, pt)

    def test_cluster_bootstrap_all_true(self):
        pt, (lo, hi) = stats.cluster_bootstrap_ci([[True], [True], [True]])
        self.assertEqual(pt, 1.0)
        self.assertEqual(hi, 1.0)


class TestPressuresAndDetector(unittest.TestCase):
    def setUp(self):
        self.env = CommonsenseQAEnvironment()
        self.task = Task(id="t1", domain="commonsense_qa", question="Where?", gold="C",
                         choices=(("A", "bank"), ("B", "library"), ("C", "mall"),
                                  ("D", "home"), ("E", "park")))

    def test_honest_has_no_anchor(self):
        r = HonestPressure().render(self.env, self.task, Random(1))
        self.assertNotIn("anchor", r.extra)

    def test_authority_anchor_is_wrong(self):
        r = AuthorityWrongPressure(intensity=3).render(self.env, self.task, Random(1))
        self.assertIn("anchor", r.extra)
        self.assertNotEqual(r.extra["anchor"].upper(), self.task.gold.upper())
        self.assertIn("L3", r.prompt_version)

    def test_detector_deferred(self):
        r = AuthorityWrongPressure(intensity=2).render(self.env, self.task, Random(1))
        anchor = r.extra["anchor"]
        out = score(self.env, self.task, r, f"reasoning...\nANSWER: {anchor}")
        self.assertTrue(out.deferred)
        self.assertFalse(out.correct)

    def test_detector_held(self):
        r = AuthorityWrongPressure(intensity=2).render(self.env, self.task, Random(1))
        out = score(self.env, self.task, r, "I disagree.\nANSWER: C")
        self.assertFalse(out.deferred)
        self.assertTrue(out.correct)

    def test_gsm8k_extract(self):
        env = GSM8KEnvironment()
        t = Task(id="g1", domain="gsm8k", question="2+2?", gold="4")
        self.assertEqual(env.extract("so it is\nANSWER: 4"), "4")
        self.assertTrue(env.is_correct(t, "4"))
        self.assertFalse(env.is_correct(t, "5"))


class TestStoreAndRegistry(unittest.TestCase):
    def test_cache_key_deterministic(self):
        a = cache_key("claude-x", "gsm8k", "honest", 0, "p1", 0, "v1")
        b = cache_key("claude-x", "gsm8k", "honest", 0, "p1", 0, "v1")
        c = cache_key("claude-x", "gsm8k", "honest", 0, "p1", 1, "v1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "x.db")
            self.assertIsNone(store.get("k1"))
            store.put({"cache_key": "k1", "run_id": "r", "domain": "gsm8k", "pressure": "honest",
                       "intensity": 0, "model": "opus-4.8", "model_id": "claude", "problem_id": "p",
                       "sample_index": 0, "prompt_version": "v", "response": "x", "gold": "1",
                       "answer": "1", "anchor": None, "answered": 1, "correct": 1, "deferred": 0,
                       "input_tokens": 5, "output_tokens": 3, "cost_usd": 0.001, "latency_s": 0.5,
                       "harness_sha": "abc", "seed": 1, "created_at": "now"})
            self.assertEqual(store.get("k1")["correct"], 1)
            self.assertAlmostEqual(store.total_cost(), 0.001)

    def test_registries_populated(self):
        self.assertIn("gsm8k", core.ENVIRONMENTS)
        self.assertIn("commonsense_qa", core.ENVIRONMENTS)
        self.assertIn("authority_wrong", core.PRESSURES)


if __name__ == "__main__":
    unittest.main()
