# Scaling report: Answer Under Pressure

*Generated from `benchmark/scale/summary.json`. 2 domains × 3 models × 3 conditions × 3 samples/item × 12 items. Cost $1.74.*

- **Provenance:** harness `0dee669` · seed 11 · samples 648 · cached 153 · errors 0

## Capability (control accuracy) by domain

| Model | gsm8k | commonsense_qa |
|---|---|---|
| `opus-4.8` | 91.7% [75.0, 100.0] | 91.7% [75.0, 100.0] |
| `sonnet-4.6` | 91.7% [75.0, 100.0] | 77.8% [52.8, 100.0] |
| `haiku-4.5` | 91.7% [75.0, 100.0] | 88.9% [75.0, 100.0] |

## Sycophancy under escalating authority (elicitation gradient)

![Pressure gradient](fig_gradient.svg)

**gsm8k** deference rate (item-clustered 95% CI):

| Model | L1 authority | L3 authority |
|---|---|---|
| `opus-4.8` | 0.0% [0.0, 0.0] | 13.9% [0.0, 33.3] |
| `sonnet-4.6` | 0.0% [0.0, 0.0] | 8.3% [0.0, 22.2] |
| `haiku-4.5` | 0.0% [0.0, 0.0] | 16.7% [8.3, 25.0] |

**commonsense_qa** deference rate (item-clustered 95% CI):

| Model | L1 authority | L3 authority |
|---|---|---|
| `opus-4.8` | 2.8% [0.0, 8.3] | 2.8% [0.0, 8.3] |
| `sonnet-4.6` | 8.3% [0.0, 25.0] | 22.2% [2.8, 44.4] |
| `haiku-4.5` | 8.3% [0.0, 16.7] | 47.2% [22.2, 75.0] |

## Model differences, FDR-corrected

Pairwise deference comparisons at the strongest authority level, with Benjamini–Hochberg correction across all comparisons (q < 0.05 = significant).

| Domain | Comparison | rate A | rate B | p | q (BH) | significant |
|---|---|---|---|---|---|---|
| gsm8k | `haiku-4.5` vs `opus-4.8` | 16.7% | 13.9% | 0.743 | 0.743 | no |
| gsm8k | `haiku-4.5` vs `sonnet-4.6` | 16.7% | 8.3% | 0.285 | 0.428 | no |
| gsm8k | `opus-4.8` vs `sonnet-4.6` | 13.9% | 8.3% | 0.453 | 0.544 | no |
| commonsense_qa | `haiku-4.5` vs `opus-4.8` | 47.2% | 2.8% | 0.000 | 0.000 | **yes** |
| commonsense_qa | `haiku-4.5` vs `sonnet-4.6` | 47.2% | 22.2% | 0.026 | 0.052 | no |
| commonsense_qa | `opus-4.8` vs `sonnet-4.6` | 2.8% | 22.2% | 0.013 | 0.038 | **yes** |

## Construct validity: does the propensity transfer across domains?

- **gsm8k** (most→least sycophantic at L3): `haiku-4.5` ‹ `opus-4.8` ‹ `sonnet-4.6`
- **commonsense_qa** (most→least sycophantic at L3): `haiku-4.5` ‹ `sonnet-4.6` ‹ `opus-4.8`

The model ranking differs across the two domains: a caution that a propensity measured in one domain may not generalize, which is exactly the kind of construct-validity check a scaled program must run.

## Limitations of this run
- Small per-cell N for a fast, cheap demonstration of the machinery; widen `--tasks` / `--samples` for publication-grade intervals.
- Significance tests use sample-level counts; a fully clustered permutation test would be the next refinement.
