import unittest

from verigrad_rl.rewards import ExactAnswerVerifier, IntegerAnswerVerifier


class RewardTests(unittest.TestCase):
    def test_exact_answer_accepts_normalized_match(self):
        result = ExactAnswerVerifier().verify("Hello world", " hello   WORLD ", {})
        self.assertTrue(result.passed)
        self.assertEqual(result.reward, 1.0)

    def test_integer_answer_rejects_non_integer(self):
        result = IntegerAnswerVerifier().verify("7", "seven", {})
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "non_integer_output")


if __name__ == "__main__":
    unittest.main()
