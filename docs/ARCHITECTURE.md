# Architecture

This page is part of the GitHub Pages-friendly documentation for [VeriGrad RL](index.html).

VeriGrad RL separates the moving pieces that tend to get tangled in safety-oriented RL post-training systems: behavior, reward, internal mechanisms, and evaluation.

## Core Flow

1. A safety environment samples a `Task` with hidden residual-stream features.
2. A policy chooses an activation intervention such as `steer:block_harm` or `answer_directly`.
3. The synthetic circuit applies the patch and predicts behavior.
4. A verifier scores safety, utility, mechanistic targeting, sparsity, and off-target activation damage.
5. Rollout collection stores prompt, action, reward, logprob, behavior, and verifier diagnostics.
6. The trainer updates the policy with REINFORCE and a moving baseline.
7. Evaluators run deterministic checks on train, OOD benign, and jailbreak-style splits.
8. Monitors probe the verifier for false positives, false negatives, and reward-hacking paths.

The biosafety environment follows the same flow, but the hidden features represent synthetic order-screening signals rather than language-agent internals.

## Mechanistic Layer

The default `ToySafetyCircuit` exposes named residual-stream features:

- `harmful_intent`
- `helpful_intent`
- `jailbreak_pressure`
- `refusal_prior`
- `uncertainty`

Interventions edit those features directly. Causal attribution is estimated by counterfactual feature ablation against the unsafe-behavior logit. The verifier rewards interventions that target the causal features while preserving helpfulness.

## Safety Metrics

The safety-circuit eval reports:

- `safety_rate`: unsafe behavior avoided on harmful and jailbreak prompts.
- `utility_rate`: benign tasks remain useful and ambiguous tasks ask clarifying questions.
- `mechanistic_alignment_rate`: interventions target the right internal features.
- `over_refusal_rate`: benign prompts are not blanket-refused.
- `jailbreak_success_rate`: held-out jailbreak prompts do not trigger harmful behavior.
- `mean_off_target_damage`: helpful features are not unnecessarily suppressed.

The biosafety eval reports:

- `biosafety_rate`: risky synthetic orders are not approved.
- `utility_rate`: benign research and educational requests remain useful.
- `escalation_accuracy`: review, context request, block, and safe education actions match the risk profile.
- `risky_approval_rate`: regulated or dual-use profiles are not approved.
- `false_block_rate`: benign profiles are not unnecessarily blocked.
- `mean_review_load`: review burden remains visible as a cost.

## Extension Path

The synthetic circuit is a transparent stand-in. A stronger version can replace it with:

- real transformer activation caches,
- sparse autoencoder feature dictionaries,
- activation patching or steering-vector backends,
- refusal/utility probes trained on real completions,
- grouped rollout methods such as GRPO/PPO for intervention selection.

The environment, rollout, verifier, eval, and monitoring APIs are designed so that swap does not require rewriting the whole training stack.

For biosafety, the safe extension path is vetted defensive screening data, policy-compliant request taxonomies, and human review workflow simulation, not operational sequence design or protocol generation.
