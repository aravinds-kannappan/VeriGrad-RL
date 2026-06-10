# VeriGrad RL

VeriGrad RL is a compact open-source lab for building and testing RL post-training workflows for text agents with verifiable rewards.

The project is designed to demonstrate the engineering surface area that makes RL research useful in practice: environment interfaces, rollout pipelines, reward/verifier separation, training loops, evaluation harnesses, monitoring, and reward-hacking checks.

## Why this exists

Modern RL for language-model agents often fails for mundane reasons before it fails for deep research reasons: brittle rewards, unclear evals, hidden train/test leakage, unstable runs, and hard-to-reproduce experiment state. VeriGrad RL keeps those pieces explicit and testable.

The default demo trains a text policy to answer arithmetic prompts using only policy-gradient updates from a verifier. It is intentionally dependency-light and runs with the Python standard library.

## Features

- Text-agent environment interface with typed task objects.
- Verifier-first rewards with exact correctness, format checks, and failure reasons.
- Rollout collection and a small REINFORCE trainer with a moving baseline.
- Greedy and sampled policy modes for train/eval separation.
- Evaluation reports for train and out-of-distribution generalization splits.
- JSONL experiment logs and reproducible config snapshots.
- Reward-hacking checks that test malformed answers, blank outputs, and verifier consistency.
- Unit tests using the standard library `unittest`.

## Quickstart

```bash
cd ~/Desktop/verigrad-rl
python3 -m verigrad_rl.cli train --episodes 50 --eval-every 25 --run-dir runs/demo
python3 -m verigrad_rl.cli eval --checkpoint runs/demo/policy.json --tasks 100
python3 -m unittest discover -s tests
```

You can also run the example script:

```bash
python3 examples/train_arithmetic.py
```

## Example output

Training writes JSONL metrics such as:

```json
{"episode": 50, "reward_mean": 0.94, "train_accuracy": 1.0, "eval_accuracy": 1.0}
```

Checkpoints are plain JSON so they can be inspected, diffed, and versioned:

```bash
python3 -m json.tool runs/demo/policy.json | head
```

## Repository layout

```text
verigrad_rl/
  envs/          Text-agent environments.
  rewards.py     Verifiers and reward contracts.
  policy.py      Feature-hashed categorical text policy.
  rollout.py     Rollout data collection.
  train.py       REINFORCE trainer.
  eval.py        Evaluation harness.
  monitors.py    Logging and reward-hacking checks.
  cli.py         Train/eval command line entrypoint.
tests/           Standard-library tests.
docs/            Architecture and extension notes.
```

See [docs/PROJECT_PITCH.md](docs/PROJECT_PITCH.md) for how to explain the project in an interview and where to take it next.

## Roadmap

- Add a PyTorch backend for sequence policies.
- Add GRPO/PPO-style grouped rollouts.
- Add sandboxed code-execution environments.
- Add richer reward-hacking probes and adversarial eval generation.
- Add W&B/MLflow adapters while preserving local JSONL as the stable baseline.

## License

MIT
