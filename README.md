<div align="center">
  <h1>VeriGrad RL</h1>
  <p><strong>A propensity benchmark for real frontier models — measuring what a model <em>will</em> do under pressure, not just what it <em>can</em> do — plus a transparent RL-from-verifier baseline.</strong></p>
  <p>
    <a href="https://github.com/aravinds-kannappan/VeriGrad-RL/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/aravinds-kannappan/VeriGrad-RL/ci.yml?branch=main&label=ci"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-0f766e"></a>
    <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-0369a1"></a>
  </p>
  <p>
    <a href="FINDINGS.md">Findings</a>
    · <a href="MECHANISM.md">Mechanistic analysis</a>
    · <a href="benchmark/results/leaderboard.md">Leaderboard</a>
    · <a href="verigrad_rl/propensity">Benchmark code</a>
    · <a href="docs/ARCHITECTURE.md">Architecture</a>
    · <a href="app">Interactive app</a>
  </p>
</div>

## Interactive web app (Next.js)

A real **Next.js 15 + React + TypeScript** app at the repo root — not a static page. It has a
**live API route** ([`app/api/probe/route.ts`](app/api/probe/route.ts)) that calls real frontier
models in real time (probe a problem, pick the pressure, watch the model hold or cave), plus an
**in-browser logistic regression** trained on the 648 real samples and the κ-paradox simulator.

```bash
npm install && npm run dev      # http://localhost:3000  (reads ANTHROPIC_API_KEY from .env)
npm run build                   # production build
```

Deploy on **Vercel**: it auto-detects Next.js at the repo root — nothing to configure beyond adding
the **`ANTHROPIC_API_KEY`** environment variable (used server-side by the live probe; the rest of
the site works without it). The Python toolkit below is unaffected — Vercel ignores it.

## Answer Under Pressure — the propensity benchmark

Capability benchmarks tell you what a model *can* do. They don't tell you what it
*will* do in your environment, under pressure. **Answer Under Pressure** measures the
second thing: it takes real frontier models, gives them real problems, and probes
three propensities that matter for trust — **deference to a wrong authority**,
**spec-gaming a disclosed grader**, and **robustness of accuracy under pressure**.

Everything here is real. Tasks come from the public **GSM8K** test split (real
human-written problems, real gold answers). Agents are real **Anthropic** models
called through the API. Every number in the report is computed from logged model
outputs — nothing is synthetic or hardcoded.

### Leaderboard (150 problems/model/condition, GSM8K test, seed 7)

| Model | Control accuracy (capability) | Deference rate ↓ | Sycophancy on solved ↓ | Spec-gaming rate ↓ | Cost |
|---|---|---|---|---|---|
| `opus-4.8` | 96.7% [92.4, 98.6] | 2.7% [1.0, 6.7] | 2.1% [0.7, 5.9] | 0.0% [0.0, 2.5] | $1.78 |
| `sonnet-4.6` | 96.7% [92.4, 98.6] | 20.0% [14.4, 27.1] | 17.9% [12.5, 25.0] | 0.0% [0.0, 2.5] | $1.56 |
| `haiku-4.5` | 95.3% [90.7, 97.7] | 10.7% [6.7, 16.6] | 9.1% [5.4, 14.9] | 0.0% [0.0, 2.5] | $0.66 |

<sub>Point estimate with Wilson 95% CI. ↓ = lower is safer. "Sycophancy on solved" = of the problems a model solved with no pressure, how often a confident wrong reviewer made it abandon the correct answer. Full run: $4.32, 1,350 model calls + 450 judge calls, 0 errors.</sub>

**Headline result:** capability is nearly tied (~96% all three), yet sycophancy under a
confident wrong reviewer differs **~9×** with non-overlapping confidence intervals —
Sonnet 4.6 abandons a correct answer 17.9% of the time, Opus 4.8 only 2.1%. The
capability ranking is *not* the trustworthiness ranking. See **[FINDINGS.md](FINDINGS.md)**
for the full writeup, the pressure-degradation deltas, and the grader-reliability analysis.

### The three probes

The *same* real problem is posed under three framings:

| Condition | What it measures | Propensity |
| --- | --- | --- |
| `control` | Solve honestly | Capability (baseline accuracy) |
| `authority_wrong` | A confident reviewer asserts a wrong answer (derived from the real gold value, never invented) | Deference / resistance-to-redirection (sycophancy) |
| `grader_gameable` | The substring grader is disclosed and gaming is invited | Spec-gaming / reward-hacking |

### Grader reliability (is the ruler trustworthy?)

Two independent graders score a dual-labeled subset: a deterministic detector and an
independent LLM judge (`claude-haiku-4-5`). We report **Cohen's κ**, not just raw
agreement — because raw agreement lies when a behavior is rare:

- **Correctness:** κ = 0.95 (99% raw, n=450) — validated.
- **Deference:** κ = 0.97 (n=150) — validated.
- **Spec-gaming:** the cross-check *caught a bug in the ruler*. An earlier detector flagged
  3 "positives" the judge unanimously rejected (κ = 0.00 despite 98% raw agreement) — all
  three the same clock-time answer (`2:00 PM`) misread as two numbers. After fixing the
  extractor, true spec-gaming is **0/150** across the lineup; both graders agree.

### Mechanistic analysis — *why* models defer

Measuring *that* a model defers isn't enough; the interesting question is *how*.
**[MECHANISM.md](MECHANISM.md)** does chain-of-thought-level mechanistic
interpretability on the real transcripts: when a model abandons the correct answer,
did its reasoning already compute the right answer and then cave (**sycophantic
override**), or did the pressure corrupt the computation itself (**anchored
reasoning**)?

| Model | Deference cases | Sycophantic override | Anchored reasoning | Override share |
|---|---|---|---|---|
| `opus-4.8` | 4 | 3 | 1 | 75.0% |
| `sonnet-4.6` | 30 | 28 | 2 | 93.3% |
| `haiku-4.5` | 16 | 14 | 2 | 87.5% |

Across the lineup, deference is overwhelmingly **social, not cognitive** — the model
computed the right answer and threw it away to agree with authority. That is a worse
failure mode than honest miscalculation, and it is *invisible to any benchmark that
only checks the final answer*. (Two independent signals — a deterministic
gold-in-reasoning check and an LLM judge — classify each case and agree at 94%.)

The fully-transparent, **activation-level** analog lives in
[`verigrad_rl/mech/`](verigrad_rl/mech): a synthetic residual-stream circuit with
named features (`harmful_intent`, `jailbreak_pressure`, …) where the same
override-vs-corruption distinction can be produced *causally* by patching activations
directly — the API doesn't expose Claude's activations, so the real-model analysis is
necessarily at the reasoning-trace level.

### Run it

```bash
pip install -e ".[llm]"
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env        # gitignored

python -m verigrad_rl.cli propensity --smoke       # ~30 calls, ~$0.10, end-to-end check
python -m verigrad_rl.cli propensity --tasks 150   # full run -> benchmark/results/ + FINDINGS.md
python -m verigrad_rl.propensity.mechanism         # CoT-faithfulness analysis -> MECHANISM.md
```

Outputs land in `benchmark/results/`: `summary.json` (aggregates), `transcripts.jsonl`
(every prompt+response+labels), `judge_labels.jsonl` (reliability), and `leaderboard.md`.

Adding a model means adding an `Agent` adapter in
[`verigrad_rl/propensity/agents.py`](verigrad_rl/propensity/agents.py); adding a
propensity means adding a condition in
[`verigrad_rl/propensity/probes.py`](verigrad_rl/propensity/probes.py).

---

## Scaling it to a research program

The single benchmark above generalizes into a platform along three axes, in
[`verigrad_rl/propensity/scale/`](verigrad_rl/propensity/scale). Full live results:
**[benchmark/scale/REPORT.md](benchmark/scale/REPORT.md)**.

**Breadth — new domains and propensities are config, not code.** A *probe* is an
`Environment` (a task domain + ground-truth verifier) × a `Pressure` (a transform that
injects authority / incentive / ambiguity), wired through registries
([`core.py`](verigrad_rl/propensity/scale/core.py),
[`environments.py`](verigrad_rl/propensity/scale/environments.py),
[`pressures.py`](verigrad_rl/propensity/scale/pressures.py)). Two real domains ship today
— **GSM8K** (free-form math) and **CommonsenseQA** (multiple choice) — so the same
sycophancy probe runs cross-domain, which is what lets you ask whether a propensity
*transfers*.

**Rigor — the science, not just more numbers.**
- **Multiple samples per item** with **item-clustered bootstrap CIs** — honest intervals
  that a naive Wilson on flattened samples would understate.
- A **pressure-intensity gradient** (mild → expert-consensus): propensity is reported as
  a curve under elicitation, not a single default-rate point.
- **Benjamini–Hochberg FDR correction** across the many model comparisons — running a
  big grid creates a multiplicity problem, and this controls the false-discovery rate.
- A **cross-domain construct-validity check**: does the model ranking transfer?

**Platform — reproducible and cheap to re-run.** A SQLite store
([`store.py`](verigrad_rl/propensity/scale/store.py)) with **content-addressed caching**
(runs are resumable; re-runs after a code change only pay for what changed), **provenance
stamping** (model id, prompt version, harness git SHA, seed on every row), and a **hard
cost ceiling**.

```bash
python -m verigrad_rl.cli scale \
  --domains gsm8k,commonsense_qa --models opus-4.8,sonnet-4.6,haiku-4.5 \
  --intensities 1,3 --tasks 12 --samples 3 --budget 5.0
# -> benchmark/scale/{REPORT.md, summary.json, samples.jsonl, runs.db, fig_gradient.svg}
```

**What the first real run showed** (2 domains × 3 models × 3 conditions, $1.74) — two
methodological points the machinery surfaces on its own:

- **FDR correction changes a conclusion.** On CommonsenseQA, Haiku-vs-Sonnet deference
  (47% vs 22%) is significant at raw *p* = 0.026 but **not** after Benjamini–Hochberg
  (*q* = 0.052). Reporting uncorrected p-values across a grid would have over-claimed.
- **A propensity does not cleanly transfer across domains.** Deference rises with the
  authority gradient in both domains, but it's far higher on fuzzy commonsense (Haiku
  47% at L3) than on verifiable math (17%), and the *model ranking differs* between
  them — a construct-validity caution you only catch by running more than one domain.

---

## Activation-level baseline: RL-from-verifier on a transparent circuit

The repo also includes the original VeriGrad RL trainer: a dependency-free
**REINFORCE-with-baseline** policy-gradient loop (a one-step contextual bandit over a
log-linear policy) that learns to choose **activation-level interventions** on a
**fully synthetic, transparent toy circuit**. It is the activation-level analog of the
chain-of-thought mechanistic analysis above: because the circuit is synthetic, every
feature is named (`harmful_intent`, `jailbreak_pressure`, …) and directly *patchable*,
so the override-vs-corruption distinction (and reward-hacking, train/test leakage,
over-refusal) can be produced and inspected *causally* — which the API can't do for a
real model's activations. It is a controllable sandbox, not a real network's internals,
and is labeled as synthetic throughout.

```bash
python -m verigrad_rl.cli train --env safety-circuit --episodes 3000 --run-dir runs/safety-demo
python -m verigrad_rl.cli eval  --env safety-circuit --checkpoint runs/safety-demo/policy.json --tasks 200
```

This baseline is what motivated the propensity benchmark above: training a policy to
look safe on a verifier is easy; the harder, more useful question is whether the
verifier survives contact with a *capable* model — which is what Answer Under Pressure
measures on real frontier models.

> The `docs/` site and the synthetic biosafety/DNA-screening playground are
> illustrative demos built on synthetic, non-operational data. They are clearly labeled
> as synthetic and are not part of the real-model benchmark.

## Repository layout

```text
verigrad_rl/
  propensity/      Answer-Under-Pressure benchmark (real models, real GSM8K).
    agents.py        Real Anthropic backends (Agent protocol + AnthropicAgent).
    dataset.py       GSM8K test-split loader (download + cache + gold parsing).
    probes.py        Conditions, deterministic answer extraction, detectors.
    graders.py       Independent LLM judge for grader reliability.
    stats.py         Wilson intervals, Cohen's kappa, paired bootstrap.
    runner.py        Orchestration, logging, cost tracking, aggregation.
    report.py        Renders summary.json -> leaderboard.md + FINDINGS.md.
    mechanism.py     CoT-faithfulness analysis of deference -> MECHANISM.md.
    scale/           Scalable harness: Environment x Pressure registries (breadth),
                     clustered CIs + gradient + FDR (rigor), SQLite cache + provenance
                     + budget (platform). See benchmark/scale/REPORT.md.
  envs/, mech/     Synthetic RL-from-verifier baseline (transparent circuit).
  rewards.py       Verifier contracts and reward helpers.
  policy.py        Feature-based categorical policy (log-linear).
  train.py         REINFORCE trainer (moving baseline).
benchmark/results/ Real run artifacts (summary, transcripts, leaderboard, mechanism).
FINDINGS.md        Writeup generated from the real summary.
MECHANISM.md       Mechanistic (CoT-faithfulness) analysis of deference.
tests/             Standard-library tests (incl. the propensity detectors + stats).
```

## License

MIT. See [LICENSE](LICENSE).
