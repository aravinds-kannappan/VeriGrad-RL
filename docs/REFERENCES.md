# References: the research behind VeriGrad

VeriGrad is built on published AI-safety research. This page maps each paper to
**where it lives in the codebase**, and is honest about the status of each link:

- **Implemented**: a real system in this repo derived from the paper.
- **Grounds**: existing code embodies the paper's idea; cited as its foundation.
- **Direction**: informs a roadmap item, not yet built.

## Mechanistic interpretability

| Paper | Status | Where |
|---|---|---|
| Conmy et al., *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023 (arXiv:2304.14997) | **Implemented** | `verigrad_rl/mech/acdc.py`: the ACDC algorithm; validated against a known answer key in `tests/test_acdc.py`, mirroring the paper's GPT-2 validation. |
| Goldowsky-Dill et al., *Localizing Model Behavior with Path Patching*, Redwood Research, 2023 (arXiv:2304.05969) | **Implemented** | `verigrad_rl/mech/circuit_graph.py`: exact edge-level path patching with unit-tested clean/corrupt invariants. |
| Rai et al., *A Practical Review of Mechanistic Interpretability for Transformer-Based LMs*, 2024 (arXiv:2407.02646) | **Grounds** | Taxonomy and framing of the `verigrad_rl/mech/` module; see [MECH_INTERP.md](MECH_INTERP.md). |

## Propensity & evaluation

| Paper | Status | Where |
|---|---|---|
| Shevlane et al., *Model Evaluation for Extreme Risks*, 2023 (arXiv:2305.15324) | **Grounds** | The capability-vs-propensity distinction the whole Answer-Under-Pressure benchmark is built on: measuring what a model *will* do, not just what it *can*. |
| MacDiarmid, Wright, Uesato et al., *Natural Emergent Misalignment from Reward Hacking in Production RL*, 2025 (arXiv:2511.18397) | **Grounds** | The `grader_gameable` / spec-gaming probe in `verigrad_rl/propensity/probes.py` and the verifiable-reward framing of the RL baseline. |
| Maia Polo et al., *Bridging Human and LLM Judgments*, 2025 (arXiv:2508.12792) | **Grounds** | The grader-reliability cross-check (Cohen's κ, dual-labeling) in `verigrad_rl/propensity/graders.py` + `stats.py`, which caught a real bug in our own ruler. |
| Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*, 2024 (arXiv:2401.05566) | **Direction** | Motivates a trigger / conditional-behavior probe (does a benign context flip the answer?), a roadmap item on the propensity side. |
| Vijayvargiya et al., *OpenAgentSafety: A Framework for Evaluating Real-World AI Agent Safety*, ICLR 2026 (arXiv:2507.06134) | **Direction** | Agent-sandbox safety as a future evaluation domain in the scalable harness. |

## RL, oversight & measurement

| Paper | Status | Where |
|---|---|---|
| Leike, Martic, Everitt et al., *AI Safety Gridworlds*, DeepMind, 2017 (arXiv:1711.09883) | **Grounds / Direction** | The safety-shaped reward design of `verigrad_rl/envs/safety_circuit.py` (safety, helpfulness preservation, off-target damage); a reward-gaming gridworld env is a roadmap item. |
| Dai et al., *Safe RLHF: Safe Reinforcement Learning from Human Feedback*, 2023 (arXiv:2310.12773) | **Grounds** | The constrained, safety-vs-utility reward shaping in the RL-from-verifier baseline (`verify()` in the safety-circuit env). |
| Brown-Cohen & Irving, *Scalable AI Safety via Doubly-Efficient Debate*, DeepMind, 2023 (arXiv:2311.14125) | **Direction** | Scalable-oversight framing for the independent judge/grader and the scaling harness. |
| Kwa, West et al. (METR), *Measuring AI Ability to Complete Long Software Tasks*, 2025 (arXiv:2503.14499) | **Grounds** | The measurement discipline behind capability-vs-propensity dissociation and the confidence-interval / FDR rigor in `verigrad_rl/propensity/scale/`. |

## Citation

If you use VeriGrad, please cite the underlying papers above as appropriate, plus
this repository (see [CITATION.cff](../CITATION.cff)).
