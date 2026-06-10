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
