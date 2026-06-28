"""Generate the site's figures from the real benchmark results.

Reads benchmark/results/summary.json and mechanism.json and emits clean,
self-contained SVG charts into docs/assets/. The figures are reproducible
artifacts of the data -- regenerate them with:

    python scripts/build_figures.py

Stdlib only (matches the repo). Colors are baked in because standalone SVGs
referenced via <img> don't inherit page CSS.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
ASSETS = ROOT / "docs" / "assets"

INK = "#0b1f33"
MUTED = "#5b6b7b"
LINE = "#dce3ea"
PAPER = "#ffffff"
TEAL = "#0f766e"
AMBER = "#b45309"
BLUE = "#0369a1"
GRID = "#eef2f6"

MODEL_COLOR = {"opus-4.8": TEAL, "sonnet-4.6": AMBER, "haiku-4.5": BLUE}
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


# --------------------------------------------------------------------------- #
# stats + svg helpers                                                         #
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def lerp(v: float, vmin: float, vmax: float, a: float, b: float) -> float:
    if vmax == vmin:
        return a
    return a + (v - vmin) / (vmax - vmin) * (b - a)


def _txt(x, y, s, *, size=13, anchor="start", fill=INK, weight=400, mono=False, opacity=1.0):
    fam = "'SFMono-Regular', ui-monospace, monospace" if mono else FONT
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'opacity="{opacity}">{s}</text>'
    )


def _svg(w: int, h: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{title}" font-family="{FONT}">'
        f'<rect width="{w}" height="{h}" fill="{PAPER}"/>{body}</svg>'
    )


def _write(name: str, svg: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / name).write_text(svg, encoding="utf-8")
    print(f"  wrote docs/assets/{name}  ({len(svg)} bytes)")


# --------------------------------------------------------------------------- #
# Figure 1: the dissociation scatter (capability vs propensity)              #
# --------------------------------------------------------------------------- #
def fig_dissociation(summary: dict) -> str:
    W, H = 760, 460
    L, R, T, B = 78, 150, 28, 64
    x0, x1, y0, y1 = L, W - R, T, H - B
    xmin, xmax = 90.0, 100.0  # control accuracy %
    ymin, ymax = 0.0, 25.0    # sycophancy %
    body: List[str] = []

    # gridlines + axes
    for gx in range(90, 101, 2):
        x = lerp(gx, xmin, xmax, x0, x1)
        body.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{GRID}"/>')
        body.append(_txt(x, y1 + 18, f"{gx}%", size=11, anchor="middle", fill=MUTED))
    for gy in range(0, 26, 5):
        y = lerp(gy, ymin, ymax, y1, y0)
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(_txt(x0 - 10, y + 4, f"{gy}%", size=11, anchor="end", fill=MUTED))
    body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{LINE}"/>')
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{LINE}"/>')

    body.append(_txt((x0 + x1) / 2, H - 14, "Capability  ·  control accuracy", size=12.5,
                     anchor="middle", fill=INK, weight=600))
    body.append(f'<text x="20" y="{(y0+y1)/2:.0f}" font-family="{FONT}" font-size="12.5" '
                f'font-weight="600" fill="{INK}" text-anchor="middle" '
                f'transform="rotate(-90 20 {(y0+y1)/2:.0f})">Propensity  ·  sycophancy on solved</text>')

    for key, m in summary["models"].items():
        acc = m["control"]["accuracy"]
        syc = m["authority_wrong"]["sycophancy_on_known"]
        ax = 100 * acc["successes"] / acc["n"]
        ay = 100 * syc["successes"] / syc["n"]
        lo, hi = wilson(syc["successes"], syc["n"])
        cx = lerp(ax, xmin, xmax, x0, x1)
        cy = lerp(ay, ymin, ymax, y1, y0)
        clo = lerp(100 * lo, ymin, ymax, y1, y0)
        chi = lerp(100 * hi, ymin, ymax, y1, y0)
        col = MODEL_COLOR.get(key, INK)
        # vertical CI whisker on the propensity axis
        body.append(f'<line x1="{cx:.1f}" y1="{chi:.1f}" x2="{cx:.1f}" y2="{clo:.1f}" '
                    f'stroke="{col}" stroke-width="2" opacity="0.5"/>')
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{col}"/>')
        body.append(_txt(cx + 14, cy + 4, f"{key}", size=13, fill=INK, weight=700))
        body.append(_txt(cx + 14, cy + 21, f"{ay:.1f}% sycophancy", size=11.5, fill=MUTED))

    body.append(_txt(x1 + 8, y0 + 6, "All three cluster", size=11.5, fill=MUTED))
    body.append(_txt(x1 + 8, y0 + 22, "on capability,", size=11.5, fill=MUTED))
    body.append(_txt(x1 + 8, y0 + 38, "but spread 9× on", size=11.5, fill=MUTED))
    body.append(_txt(x1 + 8, y0 + 54, "trustworthiness.", size=11.5, fill=MUTED, weight=700))
    return _svg(W, H, "".join(body), "Capability versus propensity dissociation")


# --------------------------------------------------------------------------- #
# Figure 2: sycophancy with Wilson CI whiskers                               #
# --------------------------------------------------------------------------- #
def fig_sycophancy(summary: dict) -> str:
    W, H = 760, 300
    L, R, T, B = 120, 70, 26, 46
    x0, x1 = L, W - R
    vmax = 30.0
    rows = sorted(
        summary["models"].items(),
        key=lambda kv: kv[1]["authority_wrong"]["sycophancy_on_known"]["successes"]
        / kv[1]["authority_wrong"]["sycophancy_on_known"]["n"],
    )
    body: List[str] = []
    for gx in range(0, 31, 5):
        x = lerp(gx, 0, vmax, x0, x1)
        body.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="{GRID}"/>')
        body.append(_txt(x, H - B + 18, f"{gx}%", size=11, anchor="middle", fill=MUTED))

    band = (H - B - T) / len(rows)
    for i, (key, m) in enumerate(rows):
        syc = m["authority_wrong"]["sycophancy_on_known"]
        p = 100 * syc["successes"] / syc["n"]
        lo, hi = wilson(syc["successes"], syc["n"])
        yc = T + band * (i + 0.5)
        bx = lerp(p, 0, vmax, x0, x1)
        col = MODEL_COLOR.get(key, INK)
        body.append(f'<rect x="{x0}" y="{yc-13:.1f}" width="{bx-x0:.1f}" height="26" rx="4" '
                    f'fill="{col}" opacity="0.88"/>')
        xlo = lerp(100 * lo, 0, vmax, x0, x1)
        xhi = lerp(100 * hi, 0, vmax, x0, x1)
        body.append(f'<line x1="{xlo:.1f}" y1="{yc:.1f}" x2="{xhi:.1f}" y2="{yc:.1f}" '
                    f'stroke="{INK}" stroke-width="1.5"/>')
        for xx in (xlo, xhi):
            body.append(f'<line x1="{xx:.1f}" y1="{yc-5:.1f}" x2="{xx:.1f}" y2="{yc+5:.1f}" '
                        f'stroke="{INK}" stroke-width="1.5"/>')
        body.append(_txt(x0 - 12, yc + 4, key, size=13, anchor="end", fill=INK, weight=650))
        body.append(_txt(xhi + 10, yc + 4, f"{p:.1f}%", size=12.5, fill=INK, weight=700, mono=True))
    body.append(_txt((x0 + x1) / 2, H - 8, "Sycophancy on solved problems  ·  Wilson 95% CI",
                     size=12, anchor="middle", fill=MUTED))
    return _svg(W, H, "".join(body), "Sycophancy with confidence intervals")


# --------------------------------------------------------------------------- #
# Figure 3: override vs anchored mechanism (stacked)                         #
# --------------------------------------------------------------------------- #
def fig_mechanism(mech: dict) -> str:
    W, H = 760, 300
    L, R, T, B = 120, 150, 26, 46
    x0, x1 = L, W - R
    rows = sorted(mech["per_model"].items(), key=lambda kv: -kv[1]["deference_n"])
    vmax = max(d["deference_n"] for _, d in rows) or 1
    body: List[str] = []
    for gx in range(0, vmax + 1, max(1, vmax // 5)):
        x = lerp(gx, 0, vmax, x0, x1)
        body.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" stroke="{GRID}"/>')
        body.append(_txt(x, H - B + 18, str(gx), size=11, anchor="middle", fill=MUTED))

    band = (H - B - T) / len(rows)
    for i, (key, d) in enumerate(rows):
        ov = d["counts"].get("sycophantic_override", 0)
        an = d["counts"].get("anchored_reasoning", 0)
        yc = T + band * (i + 0.5)
        wov = lerp(ov, 0, vmax, x0, x1) - x0
        wan = lerp(an, 0, vmax, x0, x1) - x0
        body.append(f'<rect x="{x0}" y="{yc-14:.1f}" width="{wov:.1f}" height="28" rx="4" fill="{AMBER}"/>')
        body.append(f'<rect x="{x0+wov:.1f}" y="{yc-14:.1f}" width="{wan:.1f}" height="28" rx="4" '
                    f'fill="{INK}" opacity="0.28"/>')
        body.append(_txt(x0 - 12, yc + 4, key, size=13, anchor="end", fill=INK, weight=650))
        share = 100 * ov / (ov + an) if (ov + an) else 0
        body.append(_txt(x0 + wov + wan + 12, yc + 4, f"{share:.0f}% override", size=12,
                         fill=INK, weight=700))
    # legend
    body.append(f'<rect x="{x0}" y="{T-2}" width="0" height="0"/>')
    lx = x0
    body.append(f'<rect x="{lx}" y="{H-22}" width="12" height="12" rx="2" fill="{AMBER}"/>')
    body.append(_txt(lx + 18, H - 12, "sycophantic override", size=11.5, fill=MUTED))
    body.append(f'<rect x="{lx+170}" y="{H-22}" width="12" height="12" rx="2" fill="{INK}" opacity="0.28"/>')
    body.append(_txt(lx + 188, H - 12, "anchored reasoning", size=11.5, fill=MUTED))
    return _svg(W, H, "".join(body), "Mechanism of deference: override versus anchored")


# --------------------------------------------------------------------------- #
# Figure 4: pressure degradation (control - authority accuracy, with CI)     #
# --------------------------------------------------------------------------- #
def fig_pressure(summary: dict) -> str:
    W, H = 760, 290
    L, R, T, B = 120, 70, 30, 50
    x0, x1 = L, W - R
    vmin, vmax = -5.0, 30.0
    rows = sorted(summary["models"].items(),
                  key=lambda kv: kv[1]["robustness"]["control_minus_authority_accuracy"]["point"])
    body: List[str] = []
    for gx in range(-5, 31, 5):
        x = lerp(gx, vmin, vmax, x0, x1)
        is_zero = gx == 0
        body.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}" '
                    f'stroke="{"#cbd5e1" if is_zero else GRID}" stroke-width="{1.5 if is_zero else 1}"/>')
        body.append(_txt(x, H - B + 18, f"{gx}", size=11, anchor="middle",
                         fill=INK if is_zero else MUTED, weight=700 if is_zero else 400))
    body.append(_txt(lerp(0, vmin, vmax, x0, x1), T - 10, "no change", size=10.5,
                     anchor="middle", fill=MUTED))

    band = (H - B - T) / len(rows)
    for i, (key, m) in enumerate(rows):
        d = m["robustness"]["control_minus_authority_accuracy"]
        pt = 100 * d["point"]
        lo, hi = 100 * d["ci"][0], 100 * d["ci"][1]
        yc = T + band * (i + 0.5)
        col = MODEL_COLOR.get(key, INK)
        xlo, xhi = lerp(lo, vmin, vmax, x0, x1), lerp(hi, vmin, vmax, x0, x1)
        xpt = lerp(pt, vmin, vmax, x0, x1)
        sig = lo > 0  # CI excludes zero
        body.append(f'<line x1="{xlo:.1f}" y1="{yc:.1f}" x2="{xhi:.1f}" y2="{yc:.1f}" '
                    f'stroke="{col}" stroke-width="3" opacity="0.45"/>')
        for xx in (xlo, xhi):
            body.append(f'<line x1="{xx:.1f}" y1="{yc-6:.1f}" x2="{xx:.1f}" y2="{yc+6:.1f}" '
                        f'stroke="{col}" stroke-width="2"/>')
        body.append(f'<circle cx="{xpt:.1f}" cy="{yc:.1f}" r="6.5" fill="{col}"/>')
        body.append(_txt(x0 - 12, yc + 4, key, size=13, anchor="end", fill=INK, weight=650))
        tag = "significant" if sig else "n.s."
        body.append(_txt(xhi + 10, yc + 4, f"{pt:+.1f} pts · {tag}", size=11.5,
                         fill=INK if sig else MUTED, weight=700 if sig else 500))
    body.append(_txt((x0 + x1) / 2, H - 8, "Accuracy lost under wrong-authority pressure (points, 95% CI)",
                     size=12, anchor="middle", fill=MUTED))
    return _svg(W, H, "".join(body), "Accuracy degradation under pressure")


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    mech = json.loads((RESULTS / "mechanism.json").read_text(encoding="utf-8"))
    print("Building figures from real results:")
    _write("fig_dissociation.svg", fig_dissociation(summary))
    _write("fig_sycophancy.svg", fig_sycophancy(summary))
    _write("fig_mechanism.svg", fig_mechanism(mech))
    _write("fig_pressure.svg", fig_pressure(summary))


if __name__ == "__main__":
    main()
