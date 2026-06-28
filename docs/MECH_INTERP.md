# Automated circuit discovery (ACDC + path patching)

VeriGrad ships a small, dependency-free, **white-box** implementation of automated
circuit discovery, run on a transparent safety circuit. It turns the project's
mechanistic side from "here is a hand-wired circuit" into "here is a circuit the
method *discovered* on its own, and here is the proof it is faithful."

## What it implements

- **Path patching**: `verigrad_rl/mech/circuit_graph.py`. The causal primitive from
  Goldowsky-Dill, MacLeod, Sato & Arora, *Localizing Model Behavior with Path
  Patching* (Redwood Research, arXiv:2304.05969, 2023). Each edge `(u → v)` carries
  either `u`'s clean activation or its corrupt activation; effects propagate through
  recomputation. Two invariants are unit-tested exactly: patch no edges → the clean
  run; patch all edges → the corrupt run.

- **ACDC**: `verigrad_rl/mech/acdc.py`. The algorithm from Conmy, Mavor-Parker,
  Lynch, Heimersheim & Garriga-Alonso, *Towards Automated Circuit Discovery for
  Mechanistic Interpretability* (NeurIPS 2023, arXiv:2304.14997). Walk the graph in
  reverse topological order; for each incoming edge, path-patch it to the corrupt
  activation; if the marginal KL increase is below a threshold `tau`, prune it. The
  surviving edges are the discovered circuit.

## Why a transparent substrate

The paper validates ACDC by checking it rediscovers circuits that researchers found
by hand in GPT-2 Small (e.g. 5/5 component types in the Greater-Than circuit). We do
the same thing at toy scale: because the circuit is fully white-box, it has a known
answer key (`SAFETY_DAG_GROUND_TRUTH`), so the discovery is *validated*, not just
asserted. `tests/test_acdc.py` checks the core harm-detection pathway is recovered,
that information-free edges are pruned, and that precision/recall move with `tau`
exactly as the paper's central tradeoff predicts.

## Run it

```bash
# discover the circuit behind a refuse/answer safety behavior
verigrad circuit --target safety-dag --tau 0.02
#  -> benchmark/circuits/{REPORT.md, fig_circuit.svg}

# run the same method on the RL environment's actual reward model
verigrad circuit --target toy-circuit
```

On the RL reward model (`from_toy_safety_circuit`), ACDC recovers (with no hand
labeling) that `harmful_intent` and `jailbreak_pressure` drive the harmful-completion
and safe-refusal logits, while `helpful_intent` drives the helpful-answer logit: the
harm-detection circuit, found automatically.

## How it connects to the rest of VeriGrad

The same `ToySafetyCircuit` is the reward model for the RL-from-verifier safety
environment (`verigrad_rl/envs/safety_circuit.py`), where a policy learns activation
steering interventions. Circuit discovery answers the prerequisite question that
makes such steering trustworthy: *which* components actually implement the safety
behavior. It is the white-box complement to the black-box propensity benchmark:
together, what a model does under pressure (behavior) and why (mechanism).

## Scope & honesty

This is a transparent toy circuit, not a transformer. The point is a faithful,
testable implementation of the *method* on a substrate with a known answer, wired
into a CLI and CI: a real research system brought into the codebase, not a
re-derivation on frontier weights. Porting the same `CircuitGraph` interface to real
activations (e.g. via TransformerLens) is the natural next step; see
[ROADMAP.md](../ROADMAP.md).
