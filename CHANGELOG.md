# Changelog

## 0.4.0

- Added an **Inspect AI** adapter (`verigrad_rl/integrations/inspect_task.py`): the
  propensity probes now run through the UK AISI Inspect framework against any
  provider (Anthropic, OpenAI, Google, local), with anchors seeded identically to
  the native runner so numbers are directly comparable.
- Added a harness-agnostic integration core (`integrations/_logic.py`) plus offline
  tests; `docs/INTEGRATIONS.md` documents adapters and honest interop targets.
- Added `docs/SCALING.md` describing the breadth/rigor/platform machinery and the
  axes it grows along (more providers via Inspect, domains, pressures, sharding).
- Reworked the site to read as an open-source tool: install/quickstart with
  copy-to-clipboard code blocks, a "what's in the box" feature grid, and an
  Ecosystem section separating shipped integrations from honest influences.

## 0.3.0

- Added synthetic biosafety/DNA order-screening triage environment.
- Added interactive browser playground with sliders, logits, attribution, and verifier reward.
- Added Vercel static deployment config.
- Added biosafety tests, examples, docs, and CI smoke run.

## 0.2.0

- Added `SafetyCircuitEnv` for mechanistic AI safety experiments.
- Added synthetic residual-stream circuit and activation interventions.
- Added safety metrics for utility, mechanistic alignment, over-refusal, jailbreak success, and off-target damage.
- Expanded the notebook with code, outputs, and multiple figures.
- Added GitHub Pages site assets and OSS community files.

## 0.1.0

- Added verifier-driven RL training loop.
- Added arithmetic and string-transform environments.
- Added JSONL logging, evals, tests, and CLI.
