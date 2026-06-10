import random
import unittest

from verigrad_rl.envs import ArithmeticEnv, StringTransformEnv


class EnvTests(unittest.TestCase):
    def test_arithmetic_verifier_passes_correct_answer(self):
        env = ArithmeticEnv()
        task = env.sample_task(random.Random(0))
        result = env.verify(task, task.answer)
        self.assertTrue(result.passed)

    def test_arithmetic_candidate_actions_cover_eval_answers(self):
        env = ArithmeticEnv()
        actions = set(env.candidate_actions())
        for seed in range(50):
            task = env.sample_task(random.Random(seed), split="eval")
            self.assertIn(task.answer, actions)

    def test_string_transform_has_finite_actions(self):
        env = StringTransformEnv()
        self.assertGreater(len(env.candidate_actions()), 1)


if __name__ == "__main__":
    unittest.main()
