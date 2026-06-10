# Project Pitch

This page is part of the GitHub Pages-friendly documentation for [VeriGrad RL](index.html).

VeriGrad RL is an open-source demonstration of RL infrastructure for language-model-style agents.

## What it showcases

- Turning an RL research idea into a working training loop.
- Separating rewards from verifiers so failures can be debugged.
- Building environment interfaces that can support toy policies today and larger LM agents later.
- Measuring train and held-out behavior separately.
- Logging reproducible runs with configs, metrics, summaries, and checkpoints.
- Testing reward-hacking failure modes before trusting learning curves.

## Suggested interview framing

The key point is not that arithmetic is hard. The point is that the system has the bones of a post-training platform:

- sample tasks,
- collect rollouts,
- score text actions with structured verifiers,
- update a policy,
- evaluate generalization,
- inspect failure reasons,
- check the verifier for false positives,
- save reproducible artifacts.

That is the same shape needed for more serious environments such as code repair, browser tasks, theorem proving, tool-use agents, or sandboxed research workflows.

## Honest limitations

- The default policy is a small linear baseline, not a neural LM.
- The arithmetic environment is a verifier and infrastructure demo, not a benchmark.
- Larger environments should add grouped rollouts, off-policy data storage, and model-specific logprob plumbing.

## Next impressive extension

Add a `CodeRepairEnv` where the prompt includes a broken Python function and tests, the action is a patch candidate, and the verifier runs the tests in a sandbox. That would directly extend the same reward/eval/monitoring interfaces to a more realistic agentic RL setting.
