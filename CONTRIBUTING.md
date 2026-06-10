# Contributing

Thanks for considering a contribution to VeriGrad RL.

## Local quality gate

```bash
python3 scripts/quality_gate.py
```

This runs the unit tests and a short RL training smoke test.

## Design principles

- Keep environments, policies, verifiers, and evals loosely coupled.
- Prefer structured verifier outputs over scalar-only rewards.
- Add regression tests for reward changes.
- Keep local JSONL logging working even when hosted experiment trackers are added.
- Treat reward-hacking probes as part of the core training system, not as an afterthought.

## Good first contributions

- Add a new text environment with deterministic verifier metadata.
- Add grouped rollout baselines such as GRPO-style advantage normalization.
- Add a PyTorch policy backend behind the existing policy interface.
- Add adversarial verifier probes for a specific environment.
