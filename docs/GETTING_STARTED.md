# Getting Started

## Install

```bash
git clone https://github.com/aravinds-kannappan/VeriGrad-RL.git
cd VeriGrad-RL
python3 -m unittest discover -s tests
```

VeriGrad RL currently runs with the Python standard library.

## Train the Safety Circuit

```bash
python3 -m verigrad_rl.cli train \
  --env safety-circuit \
  --episodes 3000 \
  --temperature 1.5 \
  --learning-rate 0.12 \
  --run-dir runs/safety-demo
```

## Evaluate a Checkpoint

```bash
python3 -m verigrad_rl.cli eval \
  --env safety-circuit \
  --checkpoint runs/safety-demo/policy.json \
  --tasks 200
```

## Inspect Metrics

```bash
tail -n 3 runs/safety-demo/metrics.jsonl
python3 -m json.tool runs/safety-demo/summary.json
python3 -m json.tool runs/safety-demo/policy.json | head
```

## Try Biosafety Triage

```bash
python3 -m verigrad_rl.cli train \
  --env biosafety \
  --episodes 1500 \
  --temperature 1.4 \
  --learning-rate 0.10 \
  --run-dir runs/biosafety-demo
```

The biosafety environment uses synthetic risk features and toy sequence IDs. It is a defensive triage demo, not an operational biology system.

## Regenerate Figures and Notebook

```bash
python3 scripts/generate_demo_assets.py
```

This runs the demo experiments, updates `docs/assets/*.svg`, and rewrites the notebook.
