"use client";

import { useMemo, useState } from "react";

// In-browser port of verigrad_rl/mech (path patching + ACDC) on the safety DAG.
// Drag the threshold and watch the circuit get discovered live.

type Acts = Record<string, number>;
type Edge = [string, string];
const relu = (x: number) => (x > 0 ? x : 0);

type GNode = {
  name: string;
  parents: string[];
  fn?: (a: Acts) => number;
  x: number;
  y: number;
  kind: "input" | "hidden" | "output";
};

const NODES: GNode[] = [
  { name: "harm", parents: [], x: 70, y: 70, kind: "input" },
  { name: "jailbreak", parents: [], x: 70, y: 130, kind: "input" },
  { name: "topic", parents: [], x: 70, y: 190, kind: "input" },
  { name: "refusal_cue", parents: [], x: 70, y: 250, kind: "input" },
  { name: "noise", parents: [], x: 70, y: 310, kind: "input" },
  { name: "threat", parents: ["harm", "jailbreak", "topic"], x: 255, y: 120, kind: "hidden",
    fn: (a) => relu(1.4 * a.harm + 1.0 * a.jailbreak - 0.5 * a.topic) },
  { name: "benign", parents: ["topic", "harm"], x: 255, y: 250, kind: "hidden",
    fn: (a) => relu(1.3 * a.topic - 0.8 * a.harm) },
  { name: "guard", parents: ["threat", "refusal_cue", "benign"], x: 430, y: 185, kind: "hidden",
    fn: (a) => relu(1.1 * a.threat + 0.6 * a.refusal_cue - 0.4 * a.benign) },
  { name: "refuse_logit", parents: ["guard", "benign", "noise"], x: 605, y: 120, kind: "output",
    fn: (a) => 1.5 * a.guard - 0.5 * a.benign + 0.01 * a.noise },
  { name: "answer_logit", parents: ["benign", "guard"], x: 605, y: 250, kind: "output",
    fn: (a) => 1.4 * a.benign - 1.2 * a.guard },
];

const BY_NAME = new Map(NODES.map((n) => [n.name, n]));
const ORDER = NODES.map((n) => n.name); // already topological
const OUTPUTS = ["refuse_logit", "answer_logit"];
const ALL_EDGES: Edge[] = NODES.flatMap((n) => n.parents.map((p) => [p, n.name] as Edge));
const ekey = (e: Edge) => `${e[0]}->${e[1]}`;

function run(inputs: Acts): Acts {
  const acts: Acts = {};
  for (const name of ORDER) {
    const node = BY_NAME.get(name)!;
    acts[name] = node.parents.length === 0 ? inputs[name] : node.fn!(acts);
  }
  return acts;
}

function runPatched(clean: Acts, corrupt: Acts, ablated: Set<string>): Acts {
  const corruptActs = run(corrupt);
  const patched: Acts = {};
  for (const name of ORDER) {
    const node = BY_NAME.get(name)!;
    if (node.parents.length === 0) { patched[name] = clean[name]; continue; }
    const a: Acts = {};
    for (const p of node.parents) a[p] = ablated.has(ekey([p, name])) ? corruptActs[p] : patched[p];
    patched[name] = node.fn!(a);
  }
  return patched;
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((u, v) => u + v, 0);
  return e.map((x) => x / s);
}
function kl(cleanA: Acts, otherA: Acts): number {
  const p = softmax(OUTPUTS.map((o) => cleanA[o]));
  const q = softmax(OUTPUTS.map((o) => otherA[o]));
  let t = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) t += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12));
  return t;
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const CLEAN: Acts = { harm: 0.9, jailbreak: 0.8, topic: 0.1, refusal_cue: 0.3, noise: 0.5 };
const CORRUPT: Acts = { harm: 0.1, jailbreak: 0.1, topic: 0.9, refusal_cue: 0.3, noise: 0.5 };

type Example = [Acts, Acts];
function buildDataset(n = 24): Example[] {
  const rnd = mulberry32(1);
  const j = () => (rnd() - 0.5) * 0.1;
  const out: Example[] = [];
  for (let i = 0; i < n; i++) {
    const clean: Acts = {}, corrupt: Acts = {};
    for (const k of Object.keys(CLEAN)) {
      if (Math.abs(CLEAN[k] - CORRUPT[k]) < 1e-9) { const s = CLEAN[k] + j(); clean[k] = s; corrupt[k] = s; }
      else { clean[k] = CLEAN[k] + j(); corrupt[k] = CORRUPT[k] + j(); }
    }
    out.push([clean, corrupt]);
  }
  return out;
}

function meanKl(data: Example[], ablated: Set<string>): number {
  let t = 0;
  for (const [c, d] of data) t += kl(run(c), runPatched(c, d, ablated));
  return t / data.length;
}

function runAcdc(data: Example[], tau: number) {
  const ablated = new Set<string>();
  for (let i = ALL_EDGES.length - 1; i >= 0; i--) {
    const e = ekey(ALL_EDGES[i]);
    const base = meanKl(data, ablated);
    ablated.add(e);
    const score = meanKl(data, ablated);
    if (score - base >= tau) ablated.delete(e); // keep: it mattered
  }
  const kept = ALL_EDGES.filter((e) => !ablated.has(ekey(e)));
  return { kept, ablated, faithfulness: meanKl(data, ablated) };
}

const FILL = { input: "#eef2f6", hidden: "#eff6ff", output: "#fff7ed" } as const;
const STROKE = { input: "#8694a1", hidden: "#0369a1", output: "#b45309" } as const;

export default function CircuitExplorer() {
  const [tau, setTau] = useState(0.02);
  const data = useMemo(() => buildDataset(24), []);
  const { kept, faithfulness } = useMemo(() => runAcdc(data, tau), [data, tau]);
  const keptSet = useMemo(() => new Set(kept.map(ekey)), [kept]);

  return (
    <div className="widget explorer">
      <h3>Discover the safety circuit, live</h3>
      <p className="muted-p">
        This runs the real ACDC + path-patching algorithm in your browser on the transparent safety
        DAG. Drag the threshold: a higher tau prunes more aggressively, the paper&rsquo;s precision/recall
        knob. Solid teal edges are kept; dashed are pruned.
      </p>
      <div className="controls">
        <label className="field">
          threshold tau <output>{tau.toFixed(3)}</output>
          <input type="range" min={0.001} max={0.08} step={0.001} value={tau}
            onChange={(e) => setTau(Number(e.target.value))} />
        </label>
        <div className="metrics-line">
          edges kept <b>{kept.length}/{ALL_EDGES.length}</b> · faithfulness KL{" "}
          <b>{faithfulness.toFixed(3)}</b> · sparsity <b>{Math.round((1 - kept.length / ALL_EDGES.length) * 100)}%</b>
        </div>
      </div>
      <div className="chart">
        <svg viewBox="0 0 680 360" width="100%" role="img"
          aria-label="Interactive discovered safety circuit">
          {ALL_EDGES.map((e) => {
            const a = BY_NAME.get(e[0])!, b = BY_NAME.get(e[1])!;
            const on = keptSet.has(ekey(e));
            return (
              <line key={ekey(e)} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={on ? "#14b9a0" : "#33415560"} strokeWidth={on ? 2.4 : 1.1}
                strokeDasharray={on ? "0" : "4 4"} style={{ transition: "all 0.25s ease" }} />
            );
          })}
          {NODES.map((n) => {
            const w = Math.max(54, 8 * n.name.length + 14);
            return (
              <g key={n.name}>
                <rect x={n.x - w / 2} y={n.y - 13} width={w} height={26} rx={8}
                  fill={FILL[n.kind]} stroke={STROKE[n.kind]} strokeWidth={1.4} />
                <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize={11.5} fontWeight={650} fill="#e9f2f6"
                  style={{ fill: "#0b1f33" }}>{n.name}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="muted-p" style={{ marginTop: 10 }}>
        Same code path as <code>verigrad circuit</code>: the harm-detection then guard then output
        pathway survives, while edges out of the constant inputs (refusal_cue, noise) carry no
        information and get pruned.
      </p>
    </div>
  );
}
