"use client";

import { useState } from "react";

type Metric = "sycophancy" | "deference";
type Model = { name: string; cap: number; sycophancy: number; deference: number; color: string };

const MODELS: Model[] = [
  { name: "opus-4.8", cap: 96.7, sycophancy: 2.1, deference: 2.7, color: "#0f766e" },
  { name: "sonnet-4.6", cap: 96.7, sycophancy: 17.9, deference: 20.0, color: "#b45309" },
  { name: "haiku-4.5", cap: 95.3, sycophancy: 9.1, deference: 10.7, color: "#0369a1" },
];

const W = 680, H = 380, L = 64, R = 650, T = 30, B = 320;
const xMin = 94.5, xMax = 97.5, yMin = 0, yMax = 25;
const sx = (v: number) => L + ((v - xMin) / (xMax - xMin)) * (R - L);
const sy = (v: number) => B - ((v - yMin) / (yMax - yMin)) * (B - T);

export default function ScatterPlot() {
  const [metric, setMetric] = useState<Metric>("sycophancy");
  const [hover, setHover] = useState<number | null>(null);
  const yTicks = [0, 5, 10, 15, 20, 25];
  const xTicks = [95, 96, 97];

  return (
    <figure>
      <div className="viz-head">
        <span className="viz-cap">Capability (control accuracy) vs.</span>
        <div className="seg">
          {(["sycophancy", "deference"] as Metric[]).map((m) => (
            <button key={m} className={metric === m ? "on" : ""} onClick={() => setMetric(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="figviz">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label={`Capability versus ${metric} scatter`}>
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={L} y1={sy(t)} x2={R} y2={sy(t)} stroke="#e4e9ee" strokeWidth={1} />
              <text x={L - 10} y={sy(t) + 4} textAnchor="end" fontSize={11} fill="#8694a1">{t}%</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={B + 20} textAnchor="middle" fontSize={11} fill="#8694a1">{t}%</text>
          ))}
          <line x1={L} y1={B} x2={R} y2={B} stroke="#b6c0ca" strokeWidth={1.2} />
          <text x={(L + R) / 2} y={H - 6} textAnchor="middle" fontSize={11.5} fill="#56636f">
            control accuracy (capability)
          </text>
          <text transform={`translate(16 ${(T + B) / 2}) rotate(-90)`} textAnchor="middle"
            fontSize={11.5} fill="#56636f">{metric} rate (lower is safer)</text>

          {/* the dissociation: a near-vertical band of points */}
          {MODELS.map((m, i) => {
            const x = sx(m.cap), y = sy(m[metric]);
            const on = hover === i;
            return (
              <g key={m.name} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}>
                <circle cx={x} cy={y} r={on ? 11 : 8} fill={m.color}
                  fillOpacity={on ? 1 : 0.85} stroke="#fff" strokeWidth={2}
                  style={{ transition: "r 0.15s ease" }} />
                <text x={x + 15} y={y + 4} fontSize={12.5} fontWeight={700} fill="#0b1f33">{m.name}</text>
                {on && (
                  <g>
                    <rect x={x - 92} y={y - 62} width={184} height={46} rx={8} fill="#0c1626" />
                    <text x={x} y={y - 43} textAnchor="middle" fontSize={11.5} fill="#9fb2c2">
                      capability {m.cap}%
                    </text>
                    <text x={x} y={y - 26} textAnchor="middle" fontSize={12.5} fontWeight={700} fill="#fff">
                      {metric} {m[metric]}%
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <figcaption>
        <b>Capability vs. propensity.</b> All three cluster near 96% accuracy (a near-vertical band),
        yet spread up to ~9x on {metric}. Hover a model for its numbers; toggle the axis to compare
        sycophancy and deference. The two dimensions are nearly orthogonal.
      </figcaption>
    </figure>
  );
}
