# API Reference

## Environments

`TextEnv` defines the minimal environment interface:

- `sample_task(rng, split)`
- `candidate_actions()`
- `verify(task, action)`

`SafetyCircuitEnv` implements the main AI safety demo. It samples profiles such as benign, ambiguous, harmful, jailbreak, OOD benign, and OOD jailbreak.

`BioSafetyEnv` implements the synthetic biosafety/DNA order-screening demo. It samples benign research, ambiguous orders, mock screening hits, scale-up requests, and dual-use request profiles.

## Mechanistic Core

`ToySafetyCircuit` exposes:

- `apply(snapshot, action)`
- `logits(snapshot)`
- `behavior(snapshot)`
- `attribution(snapshot)`
- `top_causal_features(snapshot)`

`ActivationSnapshot` stores named residual-stream features and supports additive intervention updates.

`ToyBioSafetyCircuit` exposes the same style of methods for synthetic biosafety triage:

- `apply(snapshot, action)`
- `logits(snapshot)`
- `behavior(snapshot)`
- `attribution(snapshot)`
- `top_causal_features(snapshot)`

## Policy

`SoftmaxTextPolicy` is a feature-based categorical policy over text actions or intervention names:

- `decide(prompt, rng, greedy=False)`
- `distribution(prompt)`
- `update(prompt, action, advantage, learning_rate)`
- `save(path)`
- `load(path)`
- `top_actions(prompt, k=5)`

## Training

`Trainer(env, policy, config).train()` runs REINFORCE with a moving reward baseline and writes:

- `config.json`
- `metrics.jsonl`
- `policy.json`
- `summary.json`

## Evaluation

`Evaluator(env, policy).run(tasks, split)` returns:

- accuracy,
- mean reward,
- failures by reason,
- reward-hacking findings,
- environment-specific extra metrics.
