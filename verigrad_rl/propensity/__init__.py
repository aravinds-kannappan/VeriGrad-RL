"""Answer-Under-Pressure: a propensity benchmark for real LLM agents.

This subpackage measures what a model *will* do under pressure, not just what it
*can* do. Real frontier models answer real GSM8K problems under three framings
(honest, wrong-authority, gameable-grader) and we score capability vs. propensity
with confidence intervals, plus an independent grader-reliability check.

Nothing here is synthetic or hardcoded: tasks come from the public GSM8K test
split, agents are real Anthropic models, and every number in the report is
computed from logged model outputs.
"""

from verigrad_rl.propensity.config import BenchmarkConfig, MODELS

__all__ = ["BenchmarkConfig", "MODELS"]
