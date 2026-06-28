"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import samples from "@/lib/samples.json";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export default function MLPlayground() {
  const NF = samples.features.length;

  const prep = useMemo(() => {
    const rows = samples.rows as number[][];
    const mean: number[] = [], std: number[] = [];
    for (let f = 0; f < NF; f++) {
      let m = 0;
      for (const r of rows) m += r[f];
      m /= rows.length;
      let v = 0;
      for (const r of rows) v += (r[f] - m) ** 2;
      mean.push(m);
      std.push(Math.sqrt(v / rows.length) || 1);
    }
    const sv = (vals: number[]) => vals.map((x, f) => (x - mean[f]) / std[f]);
    const X = rows.map((r) => sv(r.slice(0, NF)));
    const Y = rows.map((r) => r[NF]);
    return { X, Y, mean, std, sv };
  }, [NF]);

  const modelRef = useRef({ W: new Array(NF).fill(0), B: 0 });
  const [losses, setLosses] = useState<number[]>([]);
  const [coef, setCoef] = useState<number[]>(new Array(NF).fill(0));
  const [lr, setLr] = useState(0.5);
  const [epochs, setEpochs] = useState(300);
  const [training, setTraining] = useState(false);

  // predictor
  const [pModel, setPModel] = useState("haiku");
  const [pDomain, setPDomain] = useState("csqa");
  const [pIntensity, setPIntensity] = useState(3);

  function epochStep() {
    const { X, Y } = prep;
    const m = modelRef.current;
    const gw = new Array(NF).fill(0);
    let gb = 0, loss = 0;
    const n = X.length;
    for (let i = 0; i < n; i++) {
      let z = m.B;
      for (let f = 0; f < NF; f++) z += m.W[f] * X[i][f];
      const p = sigmoid(z), e = p - Y[i];
      for (let f = 0; f < NF; f++) gw[f] += e * X[i][f];
      gb += e;
      loss += -(Y[i] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
    }
    for (let f = 0; f < NF; f++) m.W[f] -= (lr * gw[f]) / n;
    m.B -= (lr * gb) / n;
    return loss / n;
  }

  function fitSync(e: number) {
    modelRef.current = { W: new Array(NF).fill(0), B: 0 };
    const arr: number[] = [];
    for (let i = 0; i < e; i++) arr.push(epochStep());
    setLosses(arr);
    setCoef([...modelRef.current.W]);
  }

  useEffect(() => {
    fitSync(300);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function train() {
    if (training) return;
    setTraining(true);
    modelRef.current = { W: new Array(NF).fill(0), B: 0 };
    const arr: number[] = [];
    let done = 0;
    const perFrame = Math.max(1, Math.ceil(epochs / 60));
    const frame = () => {
      for (let k = 0; k < perFrame && done < epochs; k++) {
        arr.push(epochStep());
        done++;
      }
      setLosses([...arr]);
      if (done < epochs) requestAnimationFrame(frame);
      else {
        setCoef([...modelRef.current.W]);
        setTraining(false);
      }
    };
    requestAnimationFrame(frame);
  }

  function predict() {
    const xs = prep.sv([pIntensity, pModel === "sonnet" ? 1 : 0, pModel === "haiku" ? 1 : 0, pDomain === "csqa" ? 1 : 0]);
    let z = modelRef.current.B;
    for (let f = 0; f < NF; f++) z += modelRef.current.W[f] * xs[f];
    return sigmoid(z);
  }

  const lossSvg = useMemo(() => {
    if (!losses.length) return null;
    const w = 480, h = 150, pl = 46, pb = 26, pt = 12, pr = 12;
    const mn = Math.min(...losses), mxRaw = Math.max(...losses);
    const mx = mxRaw === mn ? mn + 1e-6 : mxRaw;
    const pts = losses
      .map((l, i) => {
        const x = pl + (i / (losses.length - 1 || 1)) * (w - pl - pr);
        const y = pt + (1 - (l - mn) / (mx - mn)) * (h - pt - pb);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    const grid = [0, 0.5, 1].map((t) => {
      const y = pt + t * (h - pt - pb), val = mx - t * (mx - mn);
      return (
        <g key={t}>
          <line x1={pl} y1={y} x2={w - pr} y2={y} stroke="#1b2940" />
          <text x={pl - 6} y={y + 3} fill="#6f8298" fontSize={10} textAnchor="end">
            {val.toFixed(2)}
          </text>
        </g>
      );
    });
    return (
      <svg viewBox={`0 0 ${w} ${h}`} width="100%">
        {grid}
        <polyline points={pts} fill="none" stroke="#45cdab" strokeWidth={2} />
        <text x={(pl + w - pr) / 2} y={h - 4} fill="#6f8298" fontSize={10} textAnchor="middle">
          epochs → (training log-loss)
        </text>
      </svg>
    );
  }, [losses]);

  const maxAbs = Math.max(...coef.map((c) => Math.abs(c))) || 1;
  const p = predict();

  return (
    <div className="widget">
      <h3>What drives deference? Logistic regression, trained in-browser</h3>
      <p className="muted-p">
        Gradient descent fits P(defer) from four features of the real run ({samples.n} samples,{" "}
        {samples.positives} deferred). Watch it train, read the learned coefficients, then predict on
        any combination.
      </p>
      <div className="ml-grid">
        <div className="ml-left">
          <div className="controls">
            <label className="field">
              learning rate <output>{lr.toFixed(2)}</output>
              <input type="range" min={0.05} max={1.5} step={0.05} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
            </label>
            <label className="field">
              epochs <output>{epochs}</output>
              <input type="range" min={50} max={800} step={50} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
            </label>
            <button className="btn primary" onClick={train} disabled={training}>
              {training ? <span className="spinner" /> : "▶"} Train in browser
            </button>
          </div>
          <div className="chart">{lossSvg}</div>
          <div className="metrics-line">
            converged log-loss <b>{losses.length ? losses[losses.length - 1].toFixed(3) : "n/a"}</b> · trained on{" "}
            <b>{samples.n}</b> real samples
          </div>
        </div>
        <div className="ml-right">
          <h4 style={{ margin: "0 0 12px", fontSize: 14 }}>
            Learned coefficients{" "}
            <span style={{ color: "var(--faint)", fontWeight: 500, fontSize: 12 }}>standardized · + = more deference</span>
          </h4>
          {coef.map((c, f) => {
            const pos = c >= 0;
            const pct = (Math.abs(c) / maxAbs) * 50;
            return (
              <div className="coef-row" key={f}>
                <span className="coef-lab">{samples.features[f]}</span>
                <div className="coef-track">
                  <div className={`coef-bar ${pos ? "pos" : "neg"}`} style={{ width: `${pct}%`, [pos ? "left" : "right"]: "50%" } as any} />
                </div>
                <span className="coef-val">{(pos ? "+" : "") + c.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="predictor">
        <h4 style={{ margin: "0 0 12px" }}>Predict P(defer) for any combination</h4>
        <div className="controls">
          <label className="field">
            model
            <select value={pModel} onChange={(e) => setPModel(e.target.value)}>
              <option value="opus">Opus 4.8</option>
              <option value="sonnet">Sonnet 4.6</option>
              <option value="haiku">Haiku 4.5</option>
            </select>
          </label>
          <label className="field">
            domain
            <select value={pDomain} onChange={(e) => setPDomain(e.target.value)}>
              <option value="gsm8k">GSM8K (math)</option>
              <option value="csqa">CommonsenseQA</option>
            </select>
          </label>
          <label className="field">
            authority intensity <output>{pIntensity}</output>
            <input type="range" min={0} max={3} step={1} value={pIntensity} onChange={(e) => setPIntensity(Number(e.target.value))} />
          </label>
        </div>
        <div className="gauge-wrap">
          <div className="gauge">
            <div className="gauge-fill" style={{ width: `${Math.min(100, (100 * p) / 0.5)}%` }} />
          </div>
          <div className="gauge-num">{(100 * p).toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}
