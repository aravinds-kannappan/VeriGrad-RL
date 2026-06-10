# Architecture

VeriGrad RL separates the moving pieces that tend to get tangled in RL post-training systems.

## Core flow

1. An environment samples a `Task`.
2. A policy maps the task prompt to a text action.
3. A verifier scores the action and returns structured reward metadata.
4. Rollout collection stores the transition.
5. The trainer updates the policy using reward advantage.
6. Evaluators run deterministic policy checks on train and held-out splits.
7. Monitors probe the verifier for false positives and false negatives.

## Design choices

- Environments own task generation and hidden answer metadata.
- Verifiers are strict and explain failures with machine-readable reasons.
- Rollouts store enough context to debug rewards, prompts, and actions.
- Evals are deterministic by default, while training samples actions.
- Logs are JSONL so runs can be diffed and inspected without a hosted service.

## Extending to larger LM agents

Replace `SoftmaxTextPolicy` with an adapter that:

- accepts a prompt,
- samples one or more completions,
- exposes log probabilities when available,
- can be updated by a training backend or write examples for offline training.

The environment, verifier, rollout, eval, and monitoring layers do not need to change.
