"""Train the default arithmetic agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verigrad_rl.envs import ArithmeticEnv
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


def main() -> None:
    env = ArithmeticEnv()
    policy = SoftmaxTextPolicy(env.candidate_actions())
    config = TrainingConfig(episodes=50, eval_every=25, run_dir="runs/example")
    summary = Trainer(env, policy, config).train()
    print(summary)


if __name__ == "__main__":
    main()
