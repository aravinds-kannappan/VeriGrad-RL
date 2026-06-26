"""Benchmark runner: real models x conditions x real problems -> logged results.

Orchestrates the full run, logs every transcript, dual-labels a subset with the
LLM judge, and aggregates into a summary the report layer turns into a
leaderboard and a findings writeup. Token usage and cost are measured from the
API, never estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional

from verigrad_rl.propensity.agents import AnthropicAgent
from verigrad_rl.propensity.config import (
    JUDGE_MODEL_ID,
    MODELS,
    BenchmarkConfig,
    load_dotenv,
)
from verigrad_rl.propensity.dataset import sample_problems
from verigrad_rl.propensity.graders import LLMJudge
from verigrad_rl.propensity import probes
from verigrad_rl.propensity.stats import cohen_kappa, paired_bootstrap_diff, raw_agreement

JUDGE_IN_RATE, JUDGE_OUT_RATE = 1.00, 5.00  # Haiku 4.5 list price per 1M tokens


def _cost(input_tokens: int, output_tokens: int, in_rate: float, out_rate: float) -> float:
    return input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate


def _anchor_for(problem_id: int, gold: int, seed: int) -> int:
    """Deterministic, paired wrong-authority value (same for every model)."""

    return probes.wrong_anchor(gold, Random(seed * 1_000_003 + problem_id))


# --------------------------------------------------------------------------- #
# Stage 1: collect real model responses                                       #
# --------------------------------------------------------------------------- #
def _one_job(agent, model_key: str, condition: str, problem, seed: int) -> Dict[str, Any]:
    rng = Random(seed * 1_000_003 + problem.id)
    prompt, anchor = probes.build_prompt(condition, problem, rng)
    record: Dict[str, Any] = {
        "model": model_key,
        "model_id": agent.model_id,
        "condition": condition,
        "problem_id": problem.id,
        "gold": problem.gold,
        "anchor": anchor,
        "prompt": prompt,
    }
    try:
        out = agent.complete(probes.SYSTEM_PROMPT, prompt, max_tokens=1024)
        labels = probes.score(condition, out.text, problem.gold, anchor)
        record.update(
            response=out.text,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            latency_s=round(out.latency_s, 3),
            stop_reason=out.stop_reason,
            labels=asdict(labels),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - log and continue the batch
        record.update(response="", input_tokens=0, output_tokens=0, latency_s=0.0,
                      stop_reason="error", labels=None, error=f"{type(exc).__name__}: {exc}")
    return record


def collect_responses(config: BenchmarkConfig, agents: Dict[str, AnthropicAgent], problems) -> List[Dict[str, Any]]:
    jobs = [
        (agents[m], m, c, p)
        for m in config.models
        for c in config.conditions
        for p in problems
    ]
    total = len(jobs)
    print(f"[collect] {total} model calls across {len(config.models)} models x "
          f"{len(config.conditions)} conditions x {len(problems)} problems")
    records: List[Dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = [pool.submit(_one_job, a, m, c, p, config.seed) for (a, m, c, p) in jobs]
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % 25 == 0 or done == total:
                errs = sum(1 for r in records if r.get("error"))
                print(f"[collect] {done}/{total} done ({errs} errors)")
    return records


# --------------------------------------------------------------------------- #
# Stage 2: independent LLM-judge labels on a subset (for grader reliability)   #
# --------------------------------------------------------------------------- #
def judge_subset(config: BenchmarkConfig, records: List[Dict[str, Any]], judge: LLMJudge) -> List[Dict[str, Any]]:
    rng = Random(config.seed + 1)
    selected: List[Dict[str, Any]] = []
    by_cell: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("error") or r.get("labels") is None:
            continue
        by_cell.setdefault((r["model"], r["condition"]), []).append(r)
    for cell, items in by_cell.items():
        rng.shuffle(items)
        selected.extend(items[: config.judge_cap_per_cell])

    print(f"[judge] dual-labeling {len(selected)} records with {JUDGE_MODEL_ID}")
    out: List[Dict[str, Any]] = []
    done = 0

    def _do(rec: Dict[str, Any]) -> Dict[str, Any]:
        jl = judge.judge(rec["condition"], _question(rec), rec["gold"], rec["anchor"], rec["response"])
        usage = judge.last_usage
        return {
            "model": rec["model"],
            "condition": rec["condition"],
            "problem_id": rec["problem_id"],
            "det": rec["labels"],
            "judge": {"correct": jl.correct, "gamed": jl.gamed, "deferred": jl.deferred},
            "judge_tokens": [usage.input_tokens if usage else 0, usage.output_tokens if usage else 0],
        }

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = [pool.submit(_do, r) for r in selected]
        for fut in as_completed(futures):
            out.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(selected):
                print(f"[judge] {done}/{len(selected)} done")
    return out


def _question(record: Dict[str, Any]) -> str:
    # Recover the bare question (prompt has framing appended after a blank line).
    return record["prompt"].split("\n\n", 1)[0]


# --------------------------------------------------------------------------- #
# Stage 3: aggregate                                                          #
# --------------------------------------------------------------------------- #
def _rate(successes: int, n: int) -> Dict[str, int]:
    return {"successes": successes, "n": n}


def aggregate(config: BenchmarkConfig, records: List[Dict[str, Any]], judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "GSM8K test split (Cobbe et al., 2021)",
        "n_tasks": config.n_tasks,
        "seed": config.seed,
        "conditions": list(config.conditions),
        "judge_model": JUDGE_MODEL_ID,
        "models": {},
        "reliability": {},
        "totals": {},
    }

    ok = [r for r in records if not r.get("error") and r.get("labels") is not None]
    total_in = sum(r["input_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    total_cost = 0.0

    for m in config.models:
        spec = MODELS[m]
        mr = [r for r in ok if r["model"] == m]
        by_cond = {c: [r for r in mr if r["condition"] == c] for c in config.conditions}

        ctrl = by_cond.get("control", [])
        ctrl_correct = {r["problem_id"]: r["labels"]["correct"] for r in ctrl}
        ctrl_acc = sum(1 for r in ctrl if r["labels"]["correct"])

        auth = by_cond.get("authority_wrong", [])
        auth_def = sum(1 for r in auth if r["labels"]["deferred"])
        auth_acc = sum(1 for r in auth if r["labels"]["correct"])
        # Sycophancy on problems THIS model solved in control: caved to the wrong anchor?
        known = [r for r in auth if ctrl_correct.get(r["problem_id"])]
        syco_known = sum(1 for r in known if r["labels"]["deferred"])

        game = by_cond.get("grader_gameable", [])
        gamed = sum(1 for r in game if r["labels"]["gamed"])
        checker = sum(1 for r in game if r["labels"]["checker_pass"])

        # Paired robustness: control vs authority accuracy on shared problems.
        shared_ids = sorted(set(ctrl_correct) & {r["problem_id"] for r in auth})
        auth_correct = {r["problem_id"]: r["labels"]["correct"] for r in auth}
        before = [ctrl_correct[i] for i in shared_ids]
        after = [auth_correct[i] for i in shared_ids]
        delta, (dlo, dhi) = paired_bootstrap_diff(after, before, seed=config.seed)

        in_tok = sum(r["input_tokens"] for r in mr)
        out_tok = sum(r["output_tokens"] for r in mr)
        cost = _cost(in_tok, out_tok, spec.input_per_mtok, spec.output_per_mtok)
        total_cost += cost
        latencies = [r["latency_s"] for r in mr if r["latency_s"]]

        summary["models"][m] = {
            "model_id": spec.model_id,
            "control": {"accuracy": _rate(ctrl_acc, len(ctrl))},
            "authority_wrong": {
                "deference_rate": _rate(auth_def, len(auth)),
                "accuracy": _rate(auth_acc, len(auth)),
                "sycophancy_on_known": _rate(syco_known, len(known)),
            },
            "grader_gameable": {
                "spec_gaming_rate": _rate(gamed, len(game)),
                "checker_pass_rate": _rate(checker, len(game)),
            },
            "robustness": {
                "control_minus_authority_accuracy": {
                    "point": round(delta, 4),
                    "ci": [round(dlo, 4), round(dhi, 4)],
                    "n_paired": len(shared_ids),
                }
            },
            "usage": {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": round(cost, 4),
                "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "n_calls": len(mr),
                "errors": sum(1 for r in records if r["model"] == m and r.get("error")),
            },
        }

    # Grader reliability: deterministic vs LLM judge.
    judge_in = sum(j["judge_tokens"][0] for j in judged)
    judge_out = sum(j["judge_tokens"][1] for j in judged)
    judge_cost = _cost(judge_in, judge_out, JUDGE_IN_RATE, JUDGE_OUT_RATE)

    def _reliability(label: str, conds: Optional[set]) -> Dict[str, Any]:
        pairs = [
            (bool(j["det"][label]), bool(j["judge"][label]))
            for j in judged
            if conds is None or j["condition"] in conds
        ]
        det = [a for a, _ in pairs]
        jud = [b for _, b in pairs]
        return {
            "kappa": round(cohen_kappa(det, jud), 4),
            "agreement": round(raw_agreement(det, jud), 4),
            "n": len(pairs),
            "det_positives": sum(det),
            "judge_positives": sum(jud),
        }

    summary["reliability"] = {
        "correct": _reliability("correct", None),
        "deferred": _reliability("deferred", {"authority_wrong"}),
        "gamed": _reliability("gamed", {"grader_gameable"}),
        "judge_cost_usd": round(judge_cost, 4),
        "judge_calls": len(judged),
    }

    total_cost += judge_cost
    summary["totals"] = {
        "cost_usd": round(total_cost, 4),
        "input_tokens": total_in + judge_in,
        "output_tokens": total_out + judge_out,
        "n_model_calls": len(records),
        "n_judge_calls": len(judged),
        "errors": sum(1 for r in records if r.get("error")),
    }
    return summary


# --------------------------------------------------------------------------- #
# Top-level                                                                    #
# --------------------------------------------------------------------------- #
def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Create a .env file at the repo root with\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "or export it before running."
        )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    agents = {m: AnthropicAgent(m, MODELS[m].model_id) for m in config.models}
    judge = LLMJudge(AnthropicAgent("judge", JUDGE_MODEL_ID))
    problems = sample_problems(config.n_tasks, config.seed)
    print(f"[setup] {len(problems)} GSM8K problems, models={config.models}")

    started = time.monotonic()
    records = collect_responses(config, agents, problems)
    _write_jsonl(config.run_dir / "transcripts.jsonl", records)

    judged = judge_subset(config, records, judge)
    _write_jsonl(config.run_dir / "judge_labels.jsonl", judged)

    summary = aggregate(config, records, judged)
    summary["wall_clock_s"] = round(time.monotonic() - started, 1)
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] wall_clock={summary['wall_clock_s']}s  "
          f"total_cost=${summary['totals']['cost_usd']}  "
          f"errors={summary['totals']['errors']}")
    return summary


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reaggregate_from_disk(config: BenchmarkConfig) -> Dict[str, Any]:
    """Re-score logged transcripts with the current detectors and rebuild the
    summary -- no API calls. Use after changing extraction/detector logic."""

    tpath = config.run_dir / "transcripts.jsonl"
    jpath = config.run_dir / "judge_labels.jsonl"
    records = _read_jsonl(tpath)
    for r in records:
        if r.get("error") or not r.get("response"):
            continue
        labels = probes.score(r["condition"], r["response"], r["gold"], r.get("anchor"))
        r["labels"] = asdict(labels)
    _write_jsonl(tpath, records)

    det_by_key = {(r["model"], r["condition"], r["problem_id"]): r["labels"] for r in records}
    judged = _read_jsonl(jpath)
    for j in judged:
        key = (j["model"], j["condition"], j["problem_id"])
        if key in det_by_key:
            j["det"] = det_by_key[key]
    _write_jsonl(jpath, judged)

    summary = aggregate(config, records, judged)
    summary["rescored"] = True
    (config.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[rescore] rebuilt summary from {len(records)} transcripts (no API calls)")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Answer-Under-Pressure propensity benchmark")
    parser.add_argument("--models", default=",".join(MODELS.keys()),
                        help="comma-separated model keys (default: all)")
    parser.add_argument("--tasks", type=int, default=150, help="problems per model/condition")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--judge-cap", type=int, default=50,
                        help="max records per cell to dual-label for reliability")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end check (3 tasks, judge-cap 3)")
    parser.add_argument("--rescore", action="store_true",
                        help="re-aggregate logged transcripts with current detectors (no API)")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        n_tasks=3 if args.smoke else args.tasks,
        seed=args.seed,
        max_workers=args.workers,
        judge_cap_per_cell=3 if args.smoke else args.judge_cap,
    )
    if args.rescore:
        summary = reaggregate_from_disk(config)
        json.dump(summary["totals"], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    summary = run_benchmark(config)
    json.dump(summary["totals"], sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
