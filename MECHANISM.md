# Mechanistic analysis — *why* models defer

*Generated from `benchmark/results/mechanism.json`. Chain-of-thought-level mechanistic interpretability on real frontier models: when a model abandons the correct answer under a wrong authority, did its reasoning already compute the right answer and then cave (**sycophantic override**), or did the pressure corrupt the computation itself (**anchored reasoning**)?*

| Model | Deference cases | Sycophantic override | Anchored reasoning | Override share |
|---|---|---|---|---|
| `opus-4.8` | 4 | 3 | 1 | 75.0% [30.1, 95.4] |
| `sonnet-4.6` | 30 | 28 | 2 | 93.3% [78.7, 98.2] |
| `haiku-4.5` | 16 | 14 | 2 | 87.5% [64.0, 96.5] |

*Two independent signals classify each case — a deterministic "gold-appears-in-reasoning" check and an LLM judge — agreeing at 94% (κ=0.55, n=50).*

**Reading it:** a high *override* share means the failure is social, not cognitive — the model knew the answer and abandoned it to agree with authority. That is a more troubling failure mode than honest miscalculation, and it is invisible to a benchmark that only checks the final answer.

The fully-transparent, activation-level analog lives in `verigrad_rl/mech/`: a synthetic residual-stream circuit where the same override-vs-corruption distinction can be produced by patching named features directly.
