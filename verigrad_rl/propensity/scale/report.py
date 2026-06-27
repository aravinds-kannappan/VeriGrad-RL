"""Render the scale experiment into REPORT.md + a pressure-gradient figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from verigrad_rl.propensity.config import REPO_ROOT
from verigrad_rl.propensity.scale.experiment import SCALE_DIR

ASSETS = REPO_ROOT / "docs" / "assets"
MODEL_COLOR = {"opus-4.8": "#0f766e", "sonnet-4.6": "#b45309", "haiku-4.5": "#0369a1"}


def _pct(cell: Dict[str, Any]) -> str:
    if not cell:
        return "—"
    lo, hi = cell["ci"]
    return f"{100 * cell['point']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def render(run_dir: Path = SCALE_DIR) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    _gradient_figure(summary, run_dir / "fig_gradient.svg")
    if ASSETS.exists():
        (ASSETS / "fig_gradient.svg").write_text(
            (run_dir / "fig_gradient.svg").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "REPORT.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"[scale-report] wrote {run_dir/'REPORT.md'} and fig_gradient.svg")


def _markdown(s: Dict[str, Any]) -> str:
    models = s["models"]
    domains = s["domains"]
    levels = s["authority_levels"]
    L = list(map(str, levels))
    out: List[str] = []
    out.append("# Scaling report — Answer Under Pressure")
    out.append("")
    out.append(f"*Generated from `benchmark/scale/summary.json`. {len(domains)} domains × "
               f"{len(models)} models × {len(s['pressure_specs'])} conditions × "
               f"{s['k_samples']} samples/item × {s['n_tasks']} items. "
               f"Cost ${s['totals']['cost_usd']:.2f}.*")
    out.append("")
    out.append(f"- **Provenance:** harness `{s['harness_sha']}` · seed {s['seed']} · "
               f"samples {s['totals']['samples']} · cached "
               f"{s['totals'].get('cached','?')} · errors {s['totals'].get('errors',0)}")
    out.append("")

    # Capability (cross-domain)
    out.append("## Capability (control accuracy) by domain")
    out.append("")
    out.append("| Model | " + " | ".join(domains) + " |")
    out.append("|---|" + "---|" * len(domains))
    for m in models:
        cells = [_pct(s["capability"].get(d, {}).get(m, {})) for d in domains]
        out.append(f"| `{m}` | " + " | ".join(cells) + " |")
    out.append("")

    # Deference gradient per domain
    out.append("## Sycophancy under escalating authority (elicitation gradient)")
    out.append("")
    out.append("![Pressure gradient](fig_gradient.svg)")
    out.append("")
    for d in domains:
        out.append(f"**{d}** — deference rate (item-clustered 95% CI):")
        out.append("")
        out.append("| Model | " + " | ".join(f"L{l} authority" for l in levels) + " |")
        out.append("|---|" + "---|" * len(levels))
        for m in models:
            cells = [_pct(s["deference"].get(d, {}).get(str(l), {}).get(m, {})) for l in levels]
            out.append(f"| `{m}` | " + " | ".join(cells) + " |")
        out.append("")

    # FDR-corrected significance
    out.append("## Model differences, FDR-corrected")
    out.append("")
    out.append("Pairwise deference comparisons at the strongest authority level, with "
               "Benjamini–Hochberg correction across all comparisons (q < 0.05 = significant).")
    out.append("")
    out.append("| Domain | Comparison | rate A | rate B | p | q (BH) | significant |")
    out.append("|---|---|---|---|---|---|---|")
    for c in s["significance"]:
        sig = "**yes**" if c.get("significant_fdr") else "no"
        out.append(f"| {c['domain']} | `{c['model_a']}` vs `{c['model_b']}` | "
                   f"{100*c['rate_a']:.1f}% | {100*c['rate_b']:.1f}% | {c['p']:.3f} | "
                   f"{c['q']:.3f} | {sig} |")
    out.append("")

    # Construct validity
    cv = s.get("construct_validity", {})
    if cv:
        out.append("## Construct validity — does the propensity transfer across domains?")
        out.append("")
        for d, order in cv["ranking_by_domain"].items():
            out.append(f"- **{d}** (most→least sycophantic at L{cv['intensity']}): "
                       + " ‹ ".join(f"`{m}`" for m in reversed(order)))
        agree = cv["rank_agreement"]
        verb = "is identical" if agree else "differs"
        gloss = ("evidence the propensity transfers" if agree
                 else "a caution that a propensity measured in one domain may not generalize")
        out.append("")
        out.append(f"The model ranking {verb} across the two domains — {gloss}, which is exactly "
                   "the kind of construct-validity check a scaled program must run.")
        out.append("")

    out.append("## Limitations of this run")
    out.append("- Small per-cell N for a fast, cheap demonstration of the machinery; widen "
               "`--tasks` / `--samples` for publication-grade intervals.")
    out.append("- Significance tests use sample-level counts; a fully clustered permutation test "
               "would be the next refinement.")
    return "\n".join(out) + "\n"


def _gradient_figure(s: Dict[str, Any], path: Path) -> None:
    domains = s["domains"]
    models = s["models"]
    levels = [0] + s["authority_levels"]  # 0 = honest baseline (deference 0)
    panels = len(domains)
    PW, PH = 330, 250
    pad_l, pad_b, pad_t = 46, 40, 40
    W, H = panels * PW + 30, PH + 30
    xmax = max(levels) if levels else 3
    ymax = 0.0
    for d in domains:
        for m in models:
            for lv in s["authority_levels"]:
                c = s["deference"].get(d, {}).get(str(lv), {}).get(m, {})
                ymax = max(ymax, c.get("ci", [0, 0])[1] if c else 0)
    ymax = max(0.1, ymax * 1.15)
    body = [f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    def px(panel, lv):
        x0 = 20 + panel * PW + pad_l
        x1 = 20 + panel * PW + PW - 16
        return x0 + (lv / xmax) * (x1 - x0) if xmax else x0

    def py(val):
        y0, y1 = pad_t, PH - pad_b + pad_t - 8
        return y1 - (val / ymax) * (y1 - pad_t)

    for pi, d in enumerate(domains):
        x0 = 20 + pi * PW + pad_l
        x1 = 20 + pi * PW + PW - 16
        body.append(f'<text x="{(x0+x1)/2:.0f}" y="24" font-family="Inter,sans-serif" '
                    f'font-size="13" font-weight="700" fill="#0b1f33" text-anchor="middle">{d}</text>')
        for gy in (0.0, ymax / 2, ymax):
            y = py(gy)
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#eef2f6"/>')
            body.append(f'<text x="{x0-8}" y="{y+4:.1f}" font-family="Inter,sans-serif" '
                        f'font-size="10" fill="#5b6b7b" text-anchor="end">{100*gy:.0f}%</text>')
        for lv in levels:
            x = px(pi, lv)
            body.append(f'<text x="{x:.1f}" y="{PH+pad_t-8:.0f}" font-family="Inter,sans-serif" '
                        f'font-size="10" fill="#5b6b7b" text-anchor="middle">'
                        f'{"honest" if lv==0 else "L"+str(lv)}</text>')
        for m in models:
            pts = [(0, 0.0)]
            for lv in s["authority_levels"]:
                c = s["deference"].get(d, {}).get(str(lv), {}).get(m, {})
                if c:
                    pts.append((lv, c["point"]))
            col = MODEL_COLOR.get(m, "#0b1f33")
            path_d = " ".join(("M" if i == 0 else "L") + f"{px(pi,lv):.1f},{py(v):.1f}"
                              for i, (lv, v) in enumerate(pts))
            body.append(f'<path d="{path_d}" fill="none" stroke="{col}" stroke-width="2.5"/>')
            for lv, v in pts:
                body.append(f'<circle cx="{px(pi,lv):.1f}" cy="{py(v):.1f}" r="3.5" fill="{col}"/>')
    # legend
    lx = 24
    for m in models:
        body.append(f'<rect x="{lx}" y="{H-16}" width="11" height="11" rx="2" fill="{MODEL_COLOR.get(m)}"/>')
        body.append(f'<text x="{lx+16}" y="{H-7}" font-family="Inter,sans-serif" font-size="11" '
                    f'fill="#5b6b7b">{m}</text>')
        lx += 120
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'role="img" aria-label="Deference under escalating authority pressure">'
           + "".join(body) + "</svg>")
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    render()
