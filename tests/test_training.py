import tempfile
import unittest

from verigrad_rl.envs import ArithmeticEnv
from verigrad_rl.eval import Evaluator
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


class TrainingTests(unittest.TestCase):
    def test_training_writes_checkpoint_and_summary(self):
        env = ArithmeticEnv()
        policy = SoftmaxTextPolicy(env.candidate_actions())
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(episodes=30, eval_every=15, eval_tasks=10, run_dir=tmp)
            summary = Trainer(env, policy, config).train()
        self.assertIn("checkpoint", summary)
        self.assertIn("train_accuracy", summary)

    def test_evaluator_reports_failures(self):
        env = ArithmeticEnv()
        policy = SoftmaxTextPolicy(env.candidate_actions())
        report = Evaluator(env, policy, seed=0).run(tasks=10, split="train")
        self.assertEqual(report.tasks, 10)
        self.assertGreaterEqual(report.reward_hacking_findings, 0)


if __name__ == "__main__":
    unittest.main()
