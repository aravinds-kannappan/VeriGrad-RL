"use client";

import { useState } from "react";

type Props = { title?: string; code: string };

function renderLine(line: string, i: number) {
  const trimmed = line.trimStart();
  if (trimmed.startsWith("#")) {
    return <div key={i} className="cl"><span className="c">{line}</span></div>;
  }
  if (line.startsWith("$ ")) {
    return (
      <div key={i} className="cl">
        <span className="p">$</span>
        {line.slice(1)}
      </div>
    );
  }
  return <div key={i} className="cl">{line || " "}</div>;
}

export default function CodeBlock({ title = "bash", code }: Props) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable: no-op */
    }
  };

  const lines = code.replace(/\n+$/, "").split("\n");

  return (
    <div className="terminal">
      <div className="term-head">
        <span className="d" style={{ background: "#ff5f57" }} />
        <span className="d" style={{ background: "#febc2e" }} />
        <span className="d" style={{ background: "#28c840" }} />
        <span className="t">{title}</span>
        <button className="copy-btn" onClick={copy} aria-label="Copy to clipboard">
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>
      <pre>{lines.map(renderLine)}</pre>
    </div>
  );
}
