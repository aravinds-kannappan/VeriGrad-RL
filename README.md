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
  </p>
</div>

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
