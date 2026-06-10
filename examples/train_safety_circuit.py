"""Train the mechanistic safety-circuit agent."""

from verigrad_rl.envs import SafetyCircuitEnv
from verigrad_rl.policy import SoftmaxTextPolicy
from verigrad_rl.train import Trainer, TrainingConfig


def main() -> None:
    env = SafetyCircuitEnv()
    policy = SoftmaxTextPolicy(env.candidate_actions(), temperature=1.5)
    config = TrainingConfig(
        episodes=3_000,
        eval_every=300,
        eval_tasks=200,
        learning_rate=0.12,
        run_dir="runs/safety-example",
    )
    summary = Trainer(env, policy, config).train()
    print(summary)


if __name__ == "__main__":
    main()
