"""Scalable propensity-evaluation framework.

Three axes of scale on top of the single Answer-Under-Pressure benchmark:

- **Breadth**: pluggable ``Environment`` × ``Pressure`` so a new task domain or a
  new propensity is configuration, not new code (`core`, `environments`, `pressures`).
- **Rigor**: multiple samples per item with item-clustered confidence intervals, a
  pressure-intensity gradient for elicitation, and Benjamini-Hochberg FDR correction
  across the many model comparisons (`stats`, `experiment`).
- **Platform**: a SQLite results store with content-addressed caching (runs are
  resumable and re-runs are free), provenance stamping, and a hard cost ceiling
  (`store`, `experiment`).

Everything runs on real models and real datasets; nothing is synthetic.
"""

from verigrad_rl.propensity.scale.core import (
    ENVIRONMENTS,
    PRESSURES,
    Probe,
    Task,
    build_probes,
)

__all__ = ["ENVIRONMENTS", "PRESSURES", "Probe", "Task", "build_probes"]
