# Examples

Runnable examples for VeriGrad RL.

## Mechanistic Safety Circuit

```bash
python3 examples/train_safety_circuit.py
```

What it demonstrates:

- activation-level intervention actions,
- safety/utility/mechanistic verifier components,
- OOD jailbreak and benign evals,
- JSONL metrics and checkpoint output.

## Arithmetic Smoke Test

```bash
python3 examples/train_arithmetic.py
```

What it demonstrates:

- minimal verifier-driven RL,
- exact-answer rewards,
- fast CI-friendly training.

This example is intentionally simple and mainly exists to keep the core training loop easy to debug.

## Biosafety Triage

```bash
python3 examples/train_biosafety.py
```

What it demonstrates:

- synthetic DNA order-screening risk features,
- non-operational biosafety triage actions,
- safe handling of ambiguous, benign, regulated, and dual-use profiles,
- eval metrics for risky approval, false blocking, review load, and escalation accuracy.

This example uses non-sensitive mock screening fingerprints and synthetic features. It does not contain real pathogen sequences or wet-lab protocols.
