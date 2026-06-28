"""Answer-Under-Pressure as an Inspect AI task.

`Inspect <https://inspect.aisi.org.uk>`_ is the UK AI Safety Institute's open
evaluation framework (``github.com/UKGovernmentBEIS/inspect_ai``). Wrapping our
probes as Inspect ``Task``\\s means the deterministic VeriGrad detectors can run
against **any provider Inspect supports** -- not just the Anthropic lineup the
native runner targets:

.. code-block:: bash

    pip install "verigrad-rl[inspect]"

    # measure deference (sycophancy under a confident wrong reviewer)
    inspect eval verigrad_rl/integrations/inspect_task.py@deference \\
        --model anthropic/claude-sonnet-4-6 -T n_tasks=150

    # same probe, different vendor -- nothing else changes
    inspect eval verigrad_rl/integrations/inspect_task.py@deference \\
        --model openai/gpt-4o -T n_tasks=150

    # or a local model behind vLLM / Ollama
    inspect eval verigrad_rl/integrations/inspect_task.py@deference \\
        --model ollama/llama3.1 -T n_tasks=50

Each task reports a **safe-rate** (higher = safer) via Inspect's ``accuracy()``
metric, and stashes the full deterministic label record in score metadata so you
can recompute deference / spec-gaming / capability however you like.

This module is import-guarded: importing it without ``inspect-ai`` installed
raises a clear ImportError. The prompt/score logic lives in
:mod:`verigrad_rl.integrations._logic` and is unit-tested without Inspect.
"""

from __future__ import annotations

from dataclasses import asdict

from verigrad_rl.integrations._logic import (
    SAFE_LABEL,
    UNSAFE_LABEL,
    build_pressure_samples,
    label_response,
    safe_behavior,
)
from verigrad_rl.propensity.probes import SYSTEM_PROMPT

try:  # pragma: no cover - exercised only when inspect-ai is installed
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
    from inspect_ai.solver import TaskState, generate, system_message
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Inspect integration needs the `inspect-ai` package. Install it with:\n"
        '    pip install "verigrad-rl[inspect]"\n'
        "Docs: https://inspect.aisi.org.uk"
    ) from exc


@scorer(metrics=[accuracy(), stderr()])
def pressure_scorer():
    """Score one response as safe / unsafe under its framing.

    ``CORRECT`` == the model did the safe thing; the raw deterministic labels are
    attached as metadata so downstream code can read deference / gaming directly.
    """

    async def score(state: TaskState, target: Target) -> Score:
        condition = state.metadata["condition"]
        gold = int(state.metadata["gold"])
        anchor = state.metadata.get("anchor")
        text = state.output.completion or ""

        labels = label_response(condition, text, gold, anchor)
        safe = safe_behavior(condition, labels)
        behavior = SAFE_LABEL[condition] if safe else UNSAFE_LABEL[condition]

        return Score(
            value=CORRECT if safe else INCORRECT,
            answer=behavior,
            explanation=(
                f"[{condition}] model {behavior}; gold={gold}"
                + (f", wrong-anchor={anchor}" if anchor is not None else "")
            ),
            metadata={"behavior": behavior, "safe": safe, **asdict(labels)},
        )

    return score


def _pressure_task(condition: str, n_tasks: int, seed: int) -> "Task":
    samples = build_pressure_samples(condition, n_tasks, seed)
    dataset = MemoryDataset(
        [
            Sample(
                input=s.prompt,
                target=str(s.gold),
                id=f"{condition}-{s.problem_id}",
                metadata={
                    "condition": s.condition,
                    "gold": s.gold,
                    "anchor": s.anchor,
                    "problem_id": s.problem_id,
                },
            )
            for s in samples
        ]
    )
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=pressure_scorer(),
    )


@task
def capability(n_tasks: int = 150, seed: int = 7) -> "Task":
    """Control framing -> plain accuracy on GSM8K (the capability floor)."""

    return _pressure_task("control", n_tasks, seed)


@task
def deference(n_tasks: int = 150, seed: int = 7) -> "Task":
    """Authority-wrong framing -> safe-rate = 1 - deference (sycophancy probe)."""

    return _pressure_task("authority_wrong", n_tasks, seed)


@task
def spec_gaming(n_tasks: int = 150, seed: int = 7) -> "Task":
    """Gameable-grader framing -> safe-rate = 1 - spec-gaming rate."""

    return _pressure_task("grader_gameable", n_tasks, seed)


@task
def answer_under_pressure(
    condition: str = "authority_wrong", n_tasks: int = 150, seed: int = 7
) -> "Task":
    """Parametric entry point: pick any one of the three framings by name."""

    return _pressure_task(condition, n_tasks, seed)
