"""The experiment runner: where breadth, rigor, and platform meet.

Runs ``probes × models × tasks × samples`` with content-addressed caching and a cost
ceiling (platform), then aggregates with item-clustered confidence intervals, builds
the pressure-intensity gradient, and tests model differences with Benjamini-Hochberg
FDR correction (rigor), across an arbitrary set of registered domains (breadth).
"""

from __future__ import annotations

import itertools
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple

from verigrad_rl.propensity.agents import AnthropicAgent
from verigrad_rl.propensity.config import MODELS, REPO_ROOT, load_dotenv
from verigrad_rl.propensity.scale.core import Probe, build_probes, score
from verigrad_rl.propensity.scale.store import (
    Store,
    cache_key,
    harness_sha,
)
from verigrad_rl.propensity.stats import (
    benjamini_hochberg,
    cluster_bootstrap_ci,
    two_proportion_p,
)

SCALE_DIR = REPO_ROOT / "benchmark" / "scale"


@dataclass
class ScaleConfig:
    env_names: List[str] = field(default_factory=lambda: ["gsm8k", "commonsense_qa"])
    pressure_specs: List[Tuple[str, dict]] = field(default_factory=lambda: [
        ("honest", {}),
        ("authority_wrong", {"intensity": 1}),
        ("authority_wrong", {"intensity": 3}),
    ])
    model_keys: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    n_tasks: int = 12
    k_samples: int = 3
    seed: int = 11
    max_workers: int = 8
    max_tokens: int = 768
    budget_usd: float = 3.0
    run_id: str = "scale-v1"
    run_dir: Path = SCALE_DIR


def _cost(model_key: str, ti: int, to: int) -> float:
    spec = MODELS[model_key]
    return ti / 1e6 * spec.input_per_mtok + to / 1e6 * spec.output_per_mtok


def run_experiment(config: ScaleConfig) -> Dict[str, Any]:
    load_dotenv()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    store = Store(config.run_dir / "runs.db")
    sha = harness_sha()
    agents = {k: AnthropicAgent(k, MODELS[k].model_id) for k in config.model_keys}
    probes = build_probes(config.env_names, config.pressure_specs)
    from verigrad_rl.propensity.scale.core import ENVIRONMENTS
    tasks_by_env = {en: ENVIRONMENTS[en].load(config.n_tasks, config.seed) for en in config.env_names}

    # Full worklist (probe, model, task, sample_index).
    work: List[Tuple[Probe, str, Any, int]] = []
    for probe in probes:
        tasks = tasks_by_env[probe.env.name]
        for mk in config.model_keys:
            for task in tasks:
                for s in range(config.k_samples):
                    work.append((probe, mk, task, s))

    cached = sum(
        1 for (p, mk, t, s) in work
        if store.get(cache_key(MODELS[mk].model_id, p.env.name, p.pressure.name,
                               p.pressure.intensity, t.id, s, _pv(p))) is not None
    )
    todo = [w for w in work if store.get(cache_key(
        MODELS[w[1]].model_id, w[0].env.name, w[0].pressure.name,
        w[0].pressure.intensity, w[2].id, w[3], _pv(w[0]))) is None]
    print(f"[scale] {len(work)} samples total · {cached} cached · {len(todo)} to run "
          f"· budget ${config.budget_usd:.2f}")

    lock = threading.Lock()
    spent = [0.0]
    stop = threading.Event()
    counters = defaultdict(int)

    def worker(item):
        probe, mk, task, s = item
        if stop.is_set():
            counters["skipped_budget"] += 1
            return
        rng = Random(f"{task.id}|{probe.key}")
        rendered = probe.pressure.render(probe.env, task, rng)
        try:
            resp = agents[mk].complete(rendered.system, rendered.user, max_tokens=config.max_tokens)
        except Exception as exc:  # pragma: no cover
            counters["errors"] += 1
            print(f"  ! {mk} {probe.key} {task.id}#{s}: {type(exc).__name__}")
            return
        outcome = score(probe.env, task, rendered, resp.text)
        cost = _cost(mk, resp.input_tokens, resp.output_tokens)
        store.put({
            "cache_key": cache_key(resp.model_id, probe.env.name, probe.pressure.name,
                                   probe.pressure.intensity, task.id, s, _pv(probe)),
            "run_id": config.run_id, "domain": probe.env.name, "pressure": probe.pressure.name,
            "intensity": probe.pressure.intensity, "model": mk, "model_id": resp.model_id,
            "problem_id": task.id, "sample_index": s, "prompt_version": _pv(probe),
            "response": resp.text, "gold": task.gold, "answer": outcome.extra.get("answer"),
            "anchor": outcome.extra.get("anchor"),
            "answered": int(outcome.answered), "correct": int(outcome.correct),
            "deferred": int(outcome.deferred),
            "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
            "cost_usd": cost, "latency_s": resp.latency_s, "harness_sha": sha,
            "seed": config.seed, "created_at": datetime.now(timezone.utc).isoformat(),
        })
        with lock:
            spent[0] += cost
            counters["ran"] += 1
            if spent[0] >= config.budget_usd:
                stop.set()

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futs = [pool.submit(worker, w) for w in todo]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 50 == 0:
                print(f"  ... {i}/{len(todo)} ran · ${spent[0]:.2f}")

    # Re-attach run_id to cached rows so aggregation sees the full set.
    rows = _rows_for(store, config)
    summary = aggregate(rows, config, sha)
    summary["totals"].update({
        "cached": cached, "ran": counters["ran"], "errors": counters["errors"],
        "skipped_budget": counters["skipped_budget"], "store_total_cost_usd": round(store.total_cost(), 4),
    })
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    store.export_jsonl(config.run_id, config.run_dir / "samples.jsonl")
    print(f"[scale] done · ran ${spent[0]:.2f} · budget_hit={stop.is_set()} "
          f"· errors={counters['errors']}")
    return summary


def _pv(probe: Probe) -> str:
    """Prompt version string for cache keying (task-independent)."""
    from verigrad_rl.propensity.scale.core import Task
    dummy = Task(id="0", domain=probe.env.name, question="q", gold="1",
                 choices=() if probe.env.name == "gsm8k" else (("A", "x"), ("B", "y")))
    try:
        return probe.pressure.render(probe.env, dummy, Random("pv")).prompt_version
    except Exception:
        return f"{probe.pressure.name}/v1/L{probe.pressure.intensity}"


def _rows_for(store: Store, config: ScaleConfig) -> List[Dict[str, Any]]:
    # All rows matching this run's grid (covers cached rows from prior runs too).
    grid = set()
    from verigrad_rl.propensity.scale.core import ENVIRONMENTS
    for en in config.env_names:
        for t in ENVIRONMENTS[en].load(config.n_tasks, config.seed):
            grid.add((en, t.id))
    out = []
    for r in store.conn.execute("SELECT * FROM samples").fetchall():
        r = dict(r)
        if (r["domain"], r["problem_id"]) in grid and r["model"] in config.model_keys:
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Aggregation: clustered CIs, gradient, FDR-corrected significance            #
# --------------------------------------------------------------------------- #
def _clusters(rows: List[Dict[str, Any]], field_name: str) -> List[List[bool]]:
    by_item: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        by_item[r["problem_id"]].append(bool(r[field_name]))
    return list(by_item.values())


def aggregate(rows: List[Dict[str, Any]], config: ScaleConfig, sha: str) -> Dict[str, Any]:
    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    capability: Dict[str, Any] = {}
    deference: Dict[str, Any] = {}
    gradient: Dict[str, Any] = {}
    auth_levels = sorted(
        {kw.get("intensity") for name, kw in config.pressure_specs if name == "authority_wrong"})

    for en in config.env_names:
        capability[en] = {}
        deference[en] = {str(i): {} for i in auth_levels}
        gradient[en] = {}
        for mk in config.model_keys:
            hon = sel(domain=en, model=mk, pressure="honest")
            if hon:
                pt, ci = cluster_bootstrap_ci(_clusters(hon, "correct"))
                capability[en][mk] = {"point": round(pt, 4), "ci": [round(ci[0], 4), round(ci[1], 4)],
                                      "n_samples": len(hon)}
            curve = []
            # include honest as intensity 0 baseline (deference is 0 by construction; show correctness-hold instead)
            for lvl in auth_levels:
                cell = sel(domain=en, model=mk, pressure="authority_wrong", intensity=lvl)
                if not cell:
                    continue
                pt, ci = cluster_bootstrap_ci(_clusters(cell, "deferred"))
                k = sum(r["deferred"] for r in cell)
                deference[en][str(lvl)][mk] = {
                    "point": round(pt, 4), "ci": [round(ci[0], 4), round(ci[1], 4)],
                    "deferred": k, "n_samples": len(cell)}
                curve.append([lvl, round(pt, 4)])
            gradient[en][mk] = curve

    # FDR-corrected pairwise model comparisons at the strongest authority intensity.
    comparisons = []
    if auth_levels:
        strongest = max(auth_levels)
        for en in config.env_names:
            cells = {mk: sel(domain=en, model=mk, pressure="authority_wrong", intensity=strongest)
                     for mk in config.model_keys}
            cells = {mk: c for mk, c in cells.items() if c}
            for a, b in itertools.combinations(sorted(cells), 2):
                ka, na = sum(r["deferred"] for r in cells[a]), len(cells[a])
                kb, nb = sum(r["deferred"] for r in cells[b]), len(cells[b])
                comparisons.append({"domain": en, "intensity": strongest, "model_a": a, "model_b": b,
                                    "rate_a": round(ka / na, 4) if na else 0.0,
                                    "rate_b": round(kb / nb, 4) if nb else 0.0,
                                    "p": two_proportion_p(ka, na, kb, nb)})
        rejected, q = benjamini_hochberg([c["p"] for c in comparisons], alpha=0.05)
        for c, rej, qv in zip(comparisons, rejected, q):
            c["p"] = round(c["p"], 4)
            c["q"] = round(qv, 4)
            c["significant_fdr"] = bool(rej)

    # Construct validity: does the model ranking by deference transfer across domains?
    cv = _construct_validity(deference, config, auth_levels)

    totals_cost = round(sum(r["cost_usd"] for r in rows), 4)
    return {
        "run_id": config.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_sha": sha, "seed": config.seed, "n_tasks": config.n_tasks,
        "k_samples": config.k_samples, "models": config.model_keys, "domains": config.env_names,
        "pressure_specs": config.pressure_specs, "authority_levels": auth_levels,
        "capability": capability, "deference": deference, "gradient": gradient,
        "significance": comparisons, "construct_validity": cv,
        "totals": {"cost_usd": totals_cost, "samples": len(rows)},
    }


def _construct_validity(deference, config, auth_levels) -> Dict[str, Any]:
    if len(config.env_names) < 2 or not auth_levels:
        return {}
    strongest = str(max(auth_levels))
    ranks = {}
    for en in config.env_names:
        cell = deference.get(en, {}).get(strongest, {})
        ordered = sorted(cell, key=lambda m: cell[m]["point"])
        ranks[en] = ordered
    a, b = config.env_names[0], config.env_names[1]
    return {
        "intensity": int(strongest),
        "ranking_by_domain": ranks,
        "rank_agreement": ranks.get(a) == ranks.get(b),
    }
