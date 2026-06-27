"use client";

import { useState } from "react";
import problems from "@/lib/problems.json";
import { MODELS } from "@/lib/data";

type Result = {
  disabled?: boolean;
  message?: string;
  error?: string;
  detail?: string;
  response?: string;
  answer?: number | null;
  gold?: number;
  anchor?: number | null;
  correct?: boolean;
  deferred?: boolean;
  latency_s?: number;
  usage?: { input_tokens?: number; output_tokens?: number } | null;
};

export default function LiveProbe() {
  const [idx, setIdx] = useState(0);
  const [question, setQuestion] = useState(problems[0].question);
  const [gold, setGold] = useState(problems[0].gold);
  const [model, setModel] = useState("sonnet");
  const [pressure, setPressure] = useState<"none" | "authority">("authority");
  const [intensity, setIntensity] = useState(3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  function pick(i: number) {
    setIdx(i);
    setQuestion(problems[i].question);
    setGold(problems[i].gold);
    setResult(null);
  }

  async function run() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/probe", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ problem: question, gold, model, pressure, intensity }),
      });
      setResult(await res.json());
    } catch (e: any) {
      setResult({ error: "request failed", detail: String(e) });
    } finally {
      setLoading(false);
    }
  }

  const verdict = (() => {
    if (!result || result.disabled || result.error) return null;
    if (result.deferred) return { cls: "deferred", label: "Deferred to the wrong authority" };
    if (result.correct) return { cls: "held", label: "Held the correct answer" };
    return { cls: "wrong", label: "Wrong (not the planted answer)" };
  })();

  return (
    <div className="probe">
      <div className="probe-controls">
        <div className="field" style={{ width: "100%" }}>
          <span>real GSM8K problem (editable)</span>
          <textarea value={question} onChange={(e) => setQuestion(e.target.value)} />
        </div>
        <div className="probe-row">
          <label className="field" style={{ flex: "0 0 110px" }}>
            <span>gold answer</span>
            <input type="text" value={gold} onChange={(e) => setGold(e.target.value)} />
          </label>
          <label className="field">
            <span>examples</span>
            <select value={idx} onChange={(e) => pick(Number(e.target.value))}>
              {problems.map((p, i) => (
                <option key={p.id} value={i}>
                  {p.question.slice(0, 40)}…
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>pressure</span>
            <select value={pressure} onChange={(e) => setPressure(e.target.value as any)}>
              <option value="none">none (control)</option>
              <option value="authority">wrong authority</option>
            </select>
          </label>
          {pressure === "authority" && (
            <label className="field">
              <span>
                intensity <output>{intensity}</output>
              </span>
              <input
                type="range"
                min={1}
                max={3}
                step={1}
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
              />
            </label>
          )}
          <button className="btn primary" onClick={run} disabled={loading}>
            {loading ? <span className="spinner" /> : "▶"} Run live
          </button>
        </div>
      </div>

      {result && (
        <div className="result">
          {result.disabled && <p className="callout warn">{result.message}</p>}
          {result.error && (
            <p className="callout warn">
              <strong>{result.error}.</strong> {result.detail}
            </p>
          )}
          {verdict && (
            <>
              <span className={`verdict ${verdict.cls}`}>{verdict.label}</span>
              <div className="meta-line">
                model answer <b>{String(result.answer)}</b> · gold <b>{result.gold}</b>
                {result.anchor != null && <> · planted wrong answer <b>{result.anchor}</b></>} ·{" "}
                {result.latency_s?.toFixed(1)}s
                {result.usage && (
                  <> · {result.usage.input_tokens}+{result.usage.output_tokens} tok</>
                )}
              </div>
              <div className="resp-box">{result.response}</div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
