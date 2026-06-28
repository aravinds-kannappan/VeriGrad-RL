# Scaling VeriGrad to a research program

A propensity result is only worth acting on if it survives more models, more
domains, more pressure, and a re-run. This page describes the scaling machinery
that already ships and the axes it is built to grow along.

## What ships today — `verigrad_rl/propensity/scale`

The scalable runner sweeps `probes × models × tasks × samples` and is built around
three properties:

- **Breadth.** Domains are pluggable: each registered environment (`gsm8k`,
  `commonsense_qa`, …) just has to yield `(problem, gold)` pairs. Pressure is a
  spec list — `("authority_wrong", {"intensity": 1..3})` — so you add a new probe
  or a new escalation level without touching the runner.
- **Rigor.** Aggregation uses **item-clustered bootstrap CIs** (multiple samples
  per problem are not independent), builds the **pressure-intensity gradient** per
  model × domain, and corrects model-vs-model comparisons with **Benjamini–Hochberg
  FDR**. This is what flips the CommonsenseQA Haiku-vs-Sonnet result from "significant"
  (raw *p* = 0.026) to "not significant" (*q* = 0.052) — see `FINDINGS.md`.
- **Platform.** Every sample is **content-addressed** (`store.py`) by everything
  that affects its output — model id, domain, pressure, intensity, problem, sample
  index, prompt version. That buys three things at once:
  - **Resumability** — kill a run and restart; finished samples are reused.
  - **Incremental re-runs** — change one probe and you only pay to recompute what
    actually changed.
  - **Provenance** — each row carries model id, prompt version, harness git SHA,
    and seed, so any number traces back to exactly how it was produced.
- **A hard cost ceiling.** `--budget` stops the run before it crosses a dollar
  limit; the SQLite store doubles as a cost ledger from measured token usage.

```bash
verigrad scale \
  --domains gsm8k,commonsense_qa \
  --models opus-4.8,sonnet-4.6,haiku-4.5 \
  --tasks 12 --samples 3 --intensities 1,3 \
  --budget 3.0 --run-id scale-v1
# -> benchmark/scale/{REPORT.md, summary.json, samples.jsonl, runs.db, fig_gradient.svg}
```

## The axes it grows along

### 1. More providers — without rewriting the runner

The native runner targets Anthropic. The new **Inspect adapter**
(`verigrad_rl/integrations/inspect_task.py`) lifts that ceiling: because Inspect
owns the model layer, the *same* deterministic probes run against OpenAI, Google,
Mistral, or a local model behind vLLM/Ollama. Cross-vendor leaderboards become a
matter of changing `--model anthropic/... → openai/...`. See
[`INTEGRATIONS.md`](INTEGRATIONS.md).

### 2. More domains

Add an environment that yields `(problem, gold)` and register it. Math (GSM8K) and
commonsense (CommonsenseQA) ship; code, retrieval-QA, and tool-use are natural next
domains — the finding that *a propensity measured on math does not cleanly transfer
to commonsense* is the whole reason breadth matters.

### 3. More pressure types

Deference, spec-gaming, and robustness ship. The probe interface is a prompt
template plus a deterministic detector, so new propensities — flattery-seeking,
false-premise acceptance, confidence miscalibration under stakes — drop in next to
the existing three and inherit the clustered-CI + FDR pipeline for free.

### 4. More scale, safely

- **Sharding / parallelism.** Work is an embarrassingly parallel
  `(probe, model, task, sample)` list already bounded by a thread pool and a cost
  ceiling; the content-addressed store means independent shards can be merged by
  `cache_key` with no double-counting.
- **Statistical power.** `--samples k` raises samples-per-item; clustered CIs keep
  the error bars honest as `k` grows.
- **Reliability at scale.** The grader cross-check (Cohen's κ, dual-labeling) is the
  guard against a bigger run quietly drifting on a broken detector — it already
  caught one bug in our own ruler (`FINDINGS.md`).

## Non-goals

Scaling here means *more trustworthy measurement*, not bigger marketing numbers. No
synthetic tasks, no hardcoded results, and no number that is not measured from a
logged model call. A run that would cross the budget stops; a result without
provenance does not ship.
