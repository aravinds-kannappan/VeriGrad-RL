"""Real model backends.

The `Agent` protocol lets any actor plug into the benchmark; `AnthropicAgent`
calls a real Anthropic model. Calls are deliberately minimal -- no temperature,
top_p, or thinking-budget parameters, which the Opus 4.8 / Sonnet 4.6 API
surface rejects. We steer purely through the prompt, which is also the most
honest way to measure a model's default propensity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

try:  # imported lazily so the rest of the package works without the SDK
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


@dataclass(frozen=True)
class AgentResponse:
    """One completion plus the metadata we need for cost and reliability."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model_id: str
    stop_reason: str | None = None


class Agent(Protocol):
    """Anything that can answer a prompt."""

    key: str

    def complete(self, system: str, user: str, *, max_tokens: int) -> AgentResponse:
        ...


class AnthropicAgent:
    """Adapter over a single real Anthropic model."""

    def __init__(self, key: str, model_id: str, *, max_retries: int = 4) -> None:
        if anthropic is None:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required. Install with `pip install anthropic`."
            )
        self.key = key
        self.model_id = model_id
        # The SDK already retries 429/5xx with backoff; bump the ceiling for batch runs.
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def complete(self, system: str, user: str, *, max_tokens: int) -> AgentResponse:
        start = time.monotonic()
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = time.monotonic() - start
        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return AgentResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_s=latency,
            model_id=self.model_id,
            stop_reason=getattr(message, "stop_reason", None),
        )
