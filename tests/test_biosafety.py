import random
import unittest

from verigrad_rl.envs import BioSafetyEnv
from verigrad_rl.mech.biosafety import snapshot_from_slider_values


class BioSafetyTests(unittest.TestCase):
    def test_reference_actions_pass(self):
        env = BioSafetyEnv()
        rng = random.Random(11)
        for split in ("train", "eval"):
            for _ in range(50):
                task = env.sample_task(rng, split=split)
                result = env.verify(task, task.answer)
                self.assertTrue(result.passed, (task.metadata["profile"], task.answer, result.reason))

    def test_regulated_synthetic_order_is_not_approved(self):
        env = BioSafetyEnv()
        task = next(
            task
            for task in (env.sample_task(random.Random(seed)) for seed in range(200))
            if task.metadata["profile"] == "regulated_sequence"
        )
        result = env.verify(task, "approve_low_risk")
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "risky_approval")

    def test_benign_research_can_be_approved(self):
        env = BioSafetyEnv()
        task = next(
            task
            for task in (env.sample_task(random.Random(seed)) for seed in range(200))
            if task.metadata["profile"] == "benign_research"
        )
        result = env.verify(task, "approve_low_risk")
        self.assertTrue(result.passed)
        self.assertEqual(result.details["after_behavior"], "approve_order")

    def test_slider_snapshot_clamps_values(self):
        snapshot = snapshot_from_slider_values(
            sequence_risk=1.5,
            capability_risk=-0.2,
            synthesis_scale=0.4,
            benign_research_value=0.8,
            uncertainty=0.6,
        )
        self.assertEqual(snapshot.features["sequence_risk"], 1.0)
        self.assertEqual(snapshot.features["capability_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
