"""Train the synthetic biosafety triage agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verigrad_rl.envs import BioSafetyEnv
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


def main() -> None:
    env = BioSafetyEnv()
    policy = SoftmaxTextPolicy(env.candidate_actions(), temperature=1.4)
    config = TrainingConfig(
        episodes=1_500,
        eval_every=300,
        eval_tasks=120,
        learning_rate=0.10,
        run_dir="runs/biosafety-example",
    )
    summary = Trainer(env, policy, config).train()
    print(summary)


if __name__ == "__main__":
    main()
