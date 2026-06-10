import random
import unittest

from verigrad_rl.envs import SafetyCircuitEnv
from verigrad_rl.mech import ActivationSnapshot, ToySafetyCircuit


class SafetyCircuitTests(unittest.TestCase):
    def test_reference_actions_pass_verifier(self):
        env = SafetyCircuitEnv()
        rng = random.Random(5)
        for split in ("train", "eval"):
            for _ in range(50):
                task = env.sample_task(rng, split=split)
                result = env.verify(task, task.answer)
                self.assertTrue(result.passed, (task.metadata["profile"], task.answer, result.reason))

    def test_blanket_jailbreak_steering_is_penalized_on_benign_tasks(self):
        env = SafetyCircuitEnv()
        task = next(
            task
            for task in (env.sample_task(random.Random(seed)) for seed in range(100))
            if task.metadata["profile"] == "benign"
        )
        result = env.verify(task, "steer:jailbreak_detector+block_harm+preserve_helpful")
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "mechanistically_unfaithful")
        self.assertLess(result.reward, 0.70)

    def test_activation_patching_identifies_harmful_feature(self):
        circuit = ToySafetyCircuit()
        snapshot = ActivationSnapshot(
            {
                "harmful_intent": 0.9,
                "helpful_intent": 0.2,
                "jailbreak_pressure": 0.1,
                "refusal_prior": 0.1,
                "uncertainty": 0.1,
            }
        )
        attribution = circuit.attribution(snapshot)
        self.assertGreater(attribution["harmful_intent"], attribution["jailbreak_pressure"])


if __name__ == "__main__":
    unittest.main()
