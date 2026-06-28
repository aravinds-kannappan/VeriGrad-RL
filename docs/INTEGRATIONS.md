# Integrations & interoperability

VeriGrad's propensity probes are intentionally tiny and provider-neutral: a probe
is a prompt template plus a *deterministic* detector that reads the model's text
and returns a label. That design lets the **same** detectors be driven by other
people's open-source harnesses instead of only our native runner — which is the
difference between a one-off microsite and a tool that lives in the ecosystem.

## Shipped

### Inspect AI (UK AI Safety Institute) — `verigrad_rl/integrations/inspect_task.py`

[Inspect](https://inspect.aisi.org.uk) is the open evaluation framework from the
UK AI Safety Institute and the de-facto standard for safety evals. We ship a real
adapter that exposes Answer-Under-Pressure as Inspect `Task`s. Because Inspect
owns the model layer, the probes now run against **any provider Inspect supports**,
not just the Anthropic lineup the native runner targets.

```bash
pip install "verigrad-rl[inspect]"

# Deference (sycophancy under a confident wrong reviewer)
inspect eval verigrad_rl/integrations/inspect_task.py@deference \
    --model anthropic/claude-sonnet-4-6 -T n_tasks=150

# Same probe, OpenAI — nothing else changes
inspect eval verigrad_rl/integrations/inspect_task.py@deference \
    --model openai/gpt-4o -T n_tasks=150

# Same probe, a local model behind Ollama / vLLM
inspect eval verigrad_rl/integrations/inspect_task.py@deference \
    --model ollama/llama3.1 -T n_tasks=50
```

Four tasks are exposed:

| Task | Framing | `accuracy()` reports |
|---|---|---|
| `capability` | control | plain GSM8K accuracy (the capability floor) |
| `deference` | authority_wrong | safe-rate = 1 − deference (sycophancy) |
| `spec_gaming` | grader_gameable | safe-rate = 1 − spec-gaming rate |
| `answer_under_pressure` | parametric | pick any framing with `-T condition=...` |

Each score also carries the full deterministic label record in its metadata
(`deferred`, `gamed`, `checker_pass`, `numbers_on_line`, …), so you can recompute
any propensity from an Inspect `.eval` log. Anchors are seeded identically to the
native runner (`Random(seed * 1_000_003 + problem_id)`), so a number produced under
Inspect on `seed=7` matches `verigrad propensity --seed 7`.

### Real public datasets

Tasks come from **GSM8K** (Cobbe et al., 2021) and **CommonsenseQA** — downloaded
from the official sources and cached, never generated or modified. The gold answer
is parsed from the dataset's own marker. See `verigrad_rl/propensity/dataset.py`.

### Anthropic API

Models under test and the independent reliability judge are real Anthropic models
called through the API (`anthropic>=0.40`). Every dollar figure in the report is
*measured* token usage × published list price — the harness never invents a number.

## Patterned after / compatible by design

These are not bundled adapters; they are the projects whose conventions VeriGrad
deliberately mirrors so it slots into a real safety-eval workflow. Listed honestly
as influences and interop targets, not shipped integrations.

| Project | Relationship |
|---|---|
| [garak](https://github.com/NVIDIA/garak) (NVIDIA) | Probe/detector split mirrors garak's vulnerability-scanner taxonomy; our detectors are deterministic and reproducible. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI) | The `control` task is a capability baseline; the label records export cleanly to a harness-style results table. |
| [HELM](https://github.com/stanford-crfm/helm) (Stanford CRFM) | Scenario/metric separation and confidence-interval reporting follow HELM's holistic-evaluation style. |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | The activation-level RL-from-verifier baseline (`verigrad_rl/mech`) targets the same white-box, mechanistic-interpretability workflow. |
| [petri](https://github.com/anthropic-experimental/petri) (Anthropic) | Auditing-agent philosophy — probe what a model *will* do under pressure, not just what it *can* do. |

## Adding your own adapter

Everything an adapter needs is in `verigrad_rl/integrations/_logic.py` (no
third-party harness imported, fully offline-testable):

```python
from verigrad_rl.integrations import build_pressure_samples, label_response, safe_behavior

samples = build_pressure_samples("authority_wrong", n_tasks=150, seed=7)
# -> render samples[i].prompt in your harness, collect the model's text, then:
labels = label_response("authority_wrong", model_text, samples[i].gold, samples[i].anchor)
held = safe_behavior("authority_wrong", labels)   # True == did not cave
```

Translate those to your framework's `Sample` / `Score` types and you have a new
adapter. See `inspect_task.py` for a ~120-line reference.
