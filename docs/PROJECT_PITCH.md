# Project Pitch

This page is part of the GitHub Pages-friendly documentation for [VeriGrad RL](index.html).

VeriGrad RL is an open-source mechanistic interpretability and AI safety project. It trains RL policies to choose activation-level interventions, then evaluates whether those interventions are behaviorally safe, useful, and mechanistically faithful.

## What It Showcases

- Turning a safety research idea into a runnable RL training system.
- Modeling internal features explicitly with a synthetic residual stream.
- Using activation patching as the action space.
- Separating behavioral safety from mechanistic faithfulness.
- Detecting over-broad interventions that look safe but damage benign utility.
- Tracking jailbreak success and over-refusal separately.
- Logging reproducible runs with configs, metrics, summaries, checkpoints, and CI.

## Suggested Interview Framing

The key point is not that the toy circuit is a real language model. The point is that the project has the shape of a mechanistic safety platform:

- sample safety-relevant tasks,
- expose interpretable internal features,
- choose interventions,
- apply activation patches,
- score behavior and causal targeting,
- collect rollouts,
- train with RL,
- evaluate safety, utility, over-refusal, and jailbreak robustness,
- inspect reward components and failure reasons.

That is the same shape needed for real workflows using activation caches, sparse autoencoders, refusal probes, steering vectors, or circuit-level evals.

## Honest Limitations

- The default circuit is synthetic and transparent, not a real transformer.
- The baseline policy is a small linear softmax policy, not a neural LM.
- The current trainer is REINFORCE, not PPO/GRPO.
- The value is in the interfaces, eval design, and safety framing rather than raw model scale.

## Next Impressive Extension

Connect the environment to real model activations:

- cache residual streams from an open model,
- decompose activations with sparse autoencoders,
- identify refusal, harmful-intent, and helpfulness features,
- train intervention policies that steer unsafe directions while preserving benign capabilities,
- evaluate jailbreak robustness and over-refusal on held-out prompt families.
