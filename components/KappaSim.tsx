"use client";

import { useState } from "react";

function kappa(p: number, a: number) {
  const pA1 = p * a + (1 - p) * (1 - a);
  const Po = p * a * a + (1 - p) * (1 - a) * (1 - a) + p * (1 - a) * (1 - a) + (1 - p) * a * a;
  const Pe = pA1 * pA1 + (1 - pA1) * (1 - pA1);
  return { agree: Po, kappa: Pe >= 1 ? 0 : (Po - Pe) / (1 - Pe) };
}

export default function KappaSim() {
  const [rate, setRate] = useState(0.7);
  const [acc, setAcc] = useState(98);
  const { agree, kappa: k } = kappa(rate / 100, acc / 100);

  const verdict =
    rate / 100 <= 0.03 && agree > 0.9 ? (
      <>
        Raw agreement looks great, but κ has collapsed — the behavior is too rare to validate.{" "}
        <strong>This is exactly the trap that hid the spec-gaming detector bug.</strong>
      </>
    ) : k > 0.8 ? (
      <>High κ — the graders agree well beyond chance. Trustworthy.</>
    ) : (
      <>κ sits well below raw agreement — chance is doing much of the work.</>
    );

  return (
    <div className="widget">
      <h3>The κ paradox — why raw agreement lies for rare behaviors</h3>
      <p className="muted-p">
        Two independent graders label items where the true behavior occurs at a chosen base rate.
        Drag the rate down and watch Cohen&rsquo;s κ collapse while raw agreement stays high — the trap
        that hid our spec-gaming detector bug.
      </p>
      <div className="k-grid">
        <div className="controls">
          <label className="field">
            behavior base rate <output>{rate.toFixed(1)}%</output>
            <input type="range" min={0.2} max={50} step={0.1} value={rate} onChange={(e) => setRate(Number(e.target.value))} />
          </label>
          <label className="field">
            per-grader accuracy <output>{acc.toFixed(1)}%</output>
            <input type="range" min={80} max={99.9} step={0.1} value={acc} onChange={(e) => setAcc(Number(e.target.value))} />
          </label>
        </div>
        <div className="kbars">
          <div className="kb">
            <span>raw agreement</span>
            <div className="kt"><div className="kf" style={{ width: `${100 * agree}%` }} /></div>
            <b>{(100 * agree).toFixed(1)}%</b>
          </div>
          <div className="kb">
            <span>Cohen&rsquo;s κ</span>
            <div className="kt"><div className="kf amber" style={{ width: `${Math.max(0, k) * 100}%` }} /></div>
            <b>{k.toFixed(2)}</b>
          </div>
        </div>
      </div>
      <p className="callout">{verdict}</p>
    </div>
  );
}
