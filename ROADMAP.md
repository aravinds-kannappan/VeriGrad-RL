# Roadmap

## Near Term

- Add more safety profiles: prompt injection, data exfiltration, cyber misuse, persuasion, and self-harm.
- Add richer verifier probes for false-positive safety rewards.
- Add grouped rollouts for intervention selection.
- Add a small PyTorch policy backend behind the existing policy interface.
- Add benchmark JSON fixtures for reproducible eval suites.
- Expand the biosafety playground with saved scenarios and sharable URLs.

## Mechanistic Interpretability

- Connect `ToySafetyCircuit` to real transformer activation caches.
- Add sparse autoencoder feature dictionaries.
- Add steering-vector and activation-patching backends.
- Track feature-level precision and recall for interventions.
- Add causal scrubbing style tests for learned interventions.

## Product Quality

- Add hosted docs search.
- Add richer examples with saved run artifacts.
- Add experiment comparison views.
- Add optional W&B/MLflow exporters while preserving JSONL as the stable local format.

## Research Stretch Goals

- Train policies to choose interventions across multiple models.
- Add adversarial prompt generators.
- Add sleeper-agent style eval profiles.
- Add robust helpfulness preservation under distribution shift.
- Add additional defensive safety domains such as chemical synthesis triage and cyber abuse triage without operational harmful details.
