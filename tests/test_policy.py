import random
import tempfile
import unittest
from pathlib import Path

from verigrad_rl.policy import SoftmaxTextPolicy


class PolicyTests(unittest.TestCase):
    def test_policy_update_increases_chosen_action_probability(self):
        policy = SoftmaxTextPolicy(["0", "1"])
        before = policy.distribution("Solve exactly: 0 + 1")["1"]
        policy.update("Solve exactly: 0 + 1", "1", advantage=1.0, learning_rate=0.5)
        after = policy.distribution("Solve exactly: 0 + 1")["1"]
        self.assertGreater(after, before)

    def test_policy_round_trip(self):
        policy = SoftmaxTextPolicy(["0", "1"])
        policy.update("Solve exactly: 0 + 1", "1", advantage=1.0, learning_rate=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            policy.save(path)
            loaded = SoftmaxTextPolicy.load(path)
        decision = loaded.decide("Solve exactly: 0 + 1", random.Random(0), greedy=True)
        self.assertIn(decision.action, {"0", "1"})


if __name__ == "__main__":
    unittest.main()
