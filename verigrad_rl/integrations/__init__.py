"""Interoperability with the wider open-source AI-safety evaluation ecosystem.

VeriGrad's propensity probes are deliberately small and provider-neutral, so the
*same* deterministic detectors can be driven by other people's harnesses instead
of only our native runner.

Shipped here:

- :mod:`verigrad_rl.integrations.inspect_task` -- wraps Answer-Under-Pressure as
  an `Inspect AI <https://inspect.aisi.org.uk>`_ ``Task``. Inspect is the UK AI
  Safety Institute's open evaluation framework; going through it means the probes
  run against any provider Inspect supports (Anthropic, OpenAI, Google, Mistral,
  or a local model behind vLLM/Ollama), not just the Anthropic lineup.

- :mod:`verigrad_rl.integrations._logic` -- the pure prompt/score helpers shared
  by every adapter. No third-party harness needed; unit-tested offline.

The pure logic mirrors the native runner's anchor seeding exactly
(``Random(seed * 1_000_003 + problem_id)``), so a number produced under Inspect is
directly comparable to the same number from ``verigrad propensity``.
"""

from verigrad_rl.integrations._logic import (
    PressureSample,
    build_pressure_samples,
    label_response,
    safe_behavior,
)

__all__ = [
    "PressureSample",
    "build_pressure_samples",
    "label_response",
    "safe_behavior",
]
