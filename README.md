# VeriGrad RL

VeriGrad RL is an open-source mechanistic interpretability and AI safety lab for RL post-training workflows. It trains policies to choose verifiable text actions and activation-level safety interventions, then evaluates whether those interventions are behaviorally safe, useful, and mechanistically faithful.

Project site: [GitHub Pages-ready docs](https://aravinds-kannappan.github.io/VeriGrad-RL/)

The project is designed to demonstrate the engineering surface area that makes RL safety research useful in practice: environment interfaces, rollout pipelines, reward/verifier separation, mechanistic probes, training loops, evaluation harnesses, monitoring, and reward-hacking checks.

## Why this exists

Modern RL for language-model agents often fails for mundane reasons before it fails for deep research reasons: brittle rewards, unclear evals, hidden train/test leakage, unstable runs, reward hacking, and hard-to-reproduce experiment state. VeriGrad RL keeps those pieces explicit and testable.

The main demo trains a policy in a synthetic residual-stream safety circuit. The policy chooses interventions such as blocking harmful features, preserving helpful features, detecting jailbreak pressure, or asking clarifying questions. The verifier scores safety, utility retention, mechanistic targeting, sparsity, and off-target activation damage.

The smaller arithmetic demo remains as a dependency-free smoke test.

## Features

- Text-agent and mechanistic safety environment interfaces with typed task objects.
- Synthetic residual-stream circuit with named features, activation patching, and causal attribution.
- Verifier-first rewards with exact correctness, format checks, and failure reasons.
- Rollout collection and a small REINFORCE trainer with a moving baseline.
- Greedy and sampled policy modes for train/eval separation.
- Evaluation reports for train and out-of-distribution generalization splits.
- Safety metrics: safety rate, utility rate, mechanistic alignment, over-refusal, jailbreak success, and off-target damage.
- JSONL experiment logs and reproducible config snapshots.
- Reward-hacking checks that test malformed answers, blank outputs, and verifier consistency.
- Unit tests using the standard library `unittest`.

## Quickstart

```bash
cd ~/Desktop/VeriGrad-RL
python3 -m verigrad_rl.cli train --env safety-circuit --episodes 3000 --temperature 1.5 --learning-rate 0.12 --run-dir runs/safety-demo
python3 -m verigrad_rl.cli eval --env safety-circuit --checkpoint runs/safety-demo/policy.json --tasks 200
python3 -m verigrad_rl.cli train --episodes 50 --eval-every 25 --run-dir runs/demo
python3 -m unittest discover -s tests
```

You can also run the safety example script:

```bash
python3 examples/train_safety_circuit.py
```

## Example output

Training writes JSONL metrics such as:

```json
{"episode": 3000, "eval_accuracy": 1.0, "eval_extra_metrics": {"safety_rate": 1.0, "mechanistic_alignment_rate": 1.0}}
```

Checkpoints are plain JSON so they can be inspected, diffed, and versioned:

```bash
python3 -m json.tool runs/demo/policy.json | head
```

## Repository layout

```text
verigrad_rl/
  envs/          Text-agent environments.
  mech/          Synthetic residual-stream circuits and activation patching.
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

The notebook walkthrough lives at [notebooks/VeriGrad_RL_walkthrough.ipynb](notebooks/VeriGrad_RL_walkthrough.ipynb).

## Roadmap

- Add a PyTorch backend for sequence policies and real activation caches.
- Add GRPO/PPO-style grouped rollouts for intervention selection.
- Add sandboxed code-execution environments.
- Add sparse autoencoder feature dictionaries for real model activations.
- Add richer jailbreak, sleeper-agent, and over-refusal eval generation.
- Add W&B/MLflow adapters while preserving local JSONL as the stable baseline.

## License

MIT
