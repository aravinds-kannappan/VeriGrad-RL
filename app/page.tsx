import LiveProbe from "@/components/LiveProbe";
import MLPlayground from "@/components/MLPlayground";
import KappaSim from "@/components/KappaSim";
import CodeBlock from "@/components/CodeBlock";
import { LEADERBOARD, MECHANISM, GRADIENT, RELIABILITY, STATS } from "@/lib/data";

const REPO = "https://github.com/aravinds-kannappan/VeriGrad-RL";

const INSTALL = `# clone + install (editable) — Python 3.10+
git clone https://github.com/aravinds-kannappan/VeriGrad-RL
cd VeriGrad-RL
pip install -e ".[llm,inspect]"`;

const RUN = `# runs offline in seconds — verifiable-reward RL, no API key
verigrad train --episodes 400 --env arithmetic

# the propensity benchmark on real frontier models
export ANTHROPIC_API_KEY=sk-ant-...
verigrad propensity --smoke      # 3 problems, ~$0.01 — sanity check
verigrad propensity --tasks 150  # the full leaderboard run`;

const PY = `from verigrad_rl.integrations import (
    build_pressure_samples, label_response, safe_behavior,
)

# real GSM8K problems rendered under a confident, wrong reviewer
samples = build_pressure_samples("authority_wrong", n_tasks=150, seed=7)

# call any model on samples[i].prompt, then score deterministically:
labels = label_response("authority_wrong", model_text,
                        samples[0].gold, samples[0].anchor)
held = safe_behavior("authority_wrong", labels)   # True == did not cave`;

const INSPECT = `pip install "verigrad-rl[inspect]"

# the SAME probes, against any Inspect provider — one flag swaps the vendor
inspect eval verigrad_rl/integrations/inspect_task.py@deference \\
  --model anthropic/claude-sonnet-4-6
inspect eval verigrad_rl/integrations/inspect_task.py@deference \\
  --model openai/gpt-4o`;

const FEATURES = [
  {
    ic: "🎯",
    h: "Propensity benchmark",
    p: "Answer Under Pressure — deference, spec-gaming, and robustness on real GSM8K + CommonsenseQA. Measures what a model will do, not just what it can.",
  },
  {
    ic: "🔌",
    h: "Inspect AI adapter",
    p: "Run every probe through UK AISI's Inspect against Anthropic, OpenAI, Google, or a local model behind vLLM/Ollama. One flag swaps the vendor.",
  },
  {
    ic: "📈",
    h: "Scales to a program",
    p: "Content-addressed, resumable runs with a hard cost ceiling, item-clustered CIs, and Benjamini–Hochberg FDR correction across domains.",
  },
  {
    ic: "🔬",
    h: "Mechanistic analysis",
    p: "Override vs. anchored — did the model know the answer and cave, or did pressure corrupt the computation? Two independent signals agree at 94%.",
  },
  {
    ic: "✅",
    h: "We test the ruler",
    p: "Cohen's κ dual-labeling on every grader. The cross-check already caught a real bug in our own detector before it reached the report.",
  },
  {
    ic: "🧪",
    h: "RL-from-verifier baseline",
    p: "A transparent policy-gradient loop on verifiable rewards — runs offline in seconds, no GPU, every number reproducible from a seed.",
  },
];

const SHIPPED = [
  {
    name: "Inspect AI",
    by: "UK AI Safety Institute",
    desc: "Probes run as real Inspect Tasks against any provider it supports.",
    href: "https://inspect.aisi.org.uk",
  },
  {
    name: "GSM8K · CommonsenseQA",
    by: "task source",
    desc: "Real public datasets, downloaded and cached — never synthetic, never modified.",
    href: "https://github.com/openai/grade-school-math",
  },
  {
    name: "Anthropic API",
    by: "models + judge",
    desc: "Models under test and the independent reliability judge; cost is measured, not estimated.",
    href: "https://docs.anthropic.com",
  },
];

const PATTERNED = [
  {
    name: "garak",
    by: "NVIDIA",
    desc: "Probe/detector taxonomy for LLM vulnerability scanning.",
    href: "https://github.com/NVIDIA/garak",
  },
  {
    name: "lm-evaluation-harness",
    by: "EleutherAI",
    desc: "Capability-baseline + results-table conventions.",
    href: "https://github.com/EleutherAI/lm-evaluation-harness",
  },
  {
    name: "HELM",
    by: "Stanford CRFM",
    desc: "Scenario/metric separation and CI-first reporting.",
    href: "https://github.com/stanford-crfm/helm",
  },
  {
    name: "TransformerLens",
    by: "open source",
    desc: "White-box, mechanistic-interpretability workflow.",
    href: "https://github.com/TransformerLensOrg/TransformerLens",
  },
  {
    name: "petri",
    by: "Anthropic",
    desc: "Auditing-agent philosophy — probe behavior under pressure.",
    href: "https://github.com/anthropic-experimental/petri",
  },
];

export default function Home() {
  return (
    <>
      <header className="site-header">
        <div className="bar">
          <a className="brand" href="#top">
            <span className="dot" /> VeriGrad&nbsp;RL
          </a>
          <nav className="bar-links">
            <a href="#install">Install</a>
            <a href="#live">Live demo</a>
            <a href="#results">Results</a>
            <a href="#ecosystem">Ecosystem</a>
            <a href="#scaling">Scaling</a>
            <a className="ghstar" href={REPO}>★ GitHub</a>
          </nav>
        </div>
      </header>

      <section className="hero-band" id="top">
        <span className="blob b1" />
        <span className="blob b2" />
        <span className="blob b3" />
        <div className="wrap hero-inner">
          <div className="tags">
            <span className="tag">AI Safety</span>
            <span className="tag">Open Source</span>
            <span className="tag">Inspect · Next.js · Live</span>
          </div>
          <h1>Measure what a frontier model will do under pressure.</h1>
          <p className="tagline">
            VeriGrad RL is an open-source toolkit and benchmark for model{" "}
            <strong>propensities</strong> — sycophancy, spec-gaming, reasoning faithfulness. Install
            it, run the probes through <strong>Inspect</strong> against any provider, or try a live
            model in the browser below.
          </p>
          <div className="badges">
            <img src={`https://img.shields.io/github/actions/workflow/status/aravinds-kannappan/VeriGrad-RL/ci.yml?branch=main&label=CI`} alt="CI" />
            <img src="https://img.shields.io/badge/license-MIT-0f766e" alt="MIT" />
            <img src="https://img.shields.io/badge/python-3.10%2B-0369a1" alt="Python" />
            <img src="https://img.shields.io/badge/Inspect-adapter-7c3aed" alt="Inspect adapter" />
            <img src="https://img.shields.io/badge/tests-57-16a34a" alt="tests" />
            <img src={`https://img.shields.io/github/stars/aravinds-kannappan/VeriGrad-RL?style=social`} alt="stars" />
          </div>
          <div className="cta">
            <a className="btn primary" href="#install">Quickstart ↓</a>
            <a className="btn" href="#live">Try the live demo</a>
            <a className="btn" href={REPO}>View on GitHub</a>
          </div>
          <div className="statbar">
            {STATS.map((s) => (
              <div className="stat" key={s.l}>
                <b>{s.v}</b>
                <span>{s.l}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <main className="wrap">
        <section id="install">
          <p className="eyebrow">Get started · 60 seconds</p>
          <h2>Install and run it</h2>
          <p>
            A pip-installable Python package (<code>verigrad</code> CLI + library) plus this Next.js app.
            The RL baseline runs <strong>offline</strong>; the propensity benchmark calls real models and
            reads <code>ANTHROPIC_API_KEY</code> from your environment.
          </p>
          <div className="two-col">
            <CodeBlock title="install.sh" code={INSTALL} />
            <CodeBlock title="run.sh" code={RUN} />
          </div>
          <h3>Use it as a library</h3>
          <p className="muted-p">
            The probe templates and deterministic detectors are importable — render a prompt, call any
            model yourself, and score the response reproducibly.
          </p>
          <CodeBlock title="probe.py" code={PY} />
        </section>

        <section id="features">
          <p className="eyebrow">What&rsquo;s in the box</p>
          <h2>A benchmark, a baseline, and the plumbing to trust both</h2>
          <div className="features">
            {FEATURES.map((f) => (
              <div className="feat" key={f.h}>
                <div className="ic">{f.ic}</div>
                <h3>{f.h}</h3>
                <p>{f.p}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="live">
          <p className="eyebrow">Interactive · live model calls</p>
          <h2>Probe a model under pressure</h2>
          <p>
            Pick a real GSM8K problem (or edit it), choose a model and how hard a wrong &ldquo;authority&rdquo;
            pushes back, then run it against a real model. The verdict — did it hold or cave? — is computed
            from the live response.
          </p>
          <LiveProbe />
          <p className="muted-p" style={{ marginTop: 6 }}>
            Calls run server-side via a Next.js route; enable them by setting <code>ANTHROPIC_API_KEY</code>{" "}
            in the deployment. The rest of the site works without a key.
          </p>
        </section>

        <section id="results">
          <p className="eyebrow">The core finding</p>
          <h2>Capability is nearly tied. Trustworthiness is not.</h2>
          <p className="lead">Three frontier models, 150 real GSM8K problems each, under three framings.</p>
          <figure>
            <img src="/assets/fig_dissociation.svg" alt="Capability versus propensity scatter" />
            <figcaption>
              <b>Capability vs. propensity.</b> All three cluster near 96% accuracy, yet spread ~9× on
              sycophancy. The axes are nearly orthogonal.
            </figcaption>
          </figure>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Control accuracy</th>
                <th className="num">Deference ↓</th>
                <th className="num">Sycophancy on solved ↓</th>
                <th className="num">Spec-gaming ↓</th>
                <th className="num">Cost</th>
              </tr>
            </thead>
            <tbody>
              {LEADERBOARD.map((r) => (
                <tr key={r.model}>
                  <td><code>{r.model}</code></td>
                  <td className="num">{r.control}</td>
                  <td className="num">{r.deference}</td>
                  <td className="num">
                    <span className={r.tone ?? ""}>{r.sycophancy}</span>{" "}
                    <span className="ci">{r.sycophancyCI}</span>
                  </td>
                  <td className="num">{r.spec}</td>
                  <td className="num">{r.cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="callout">
            <strong>Headline.</strong> Sonnet 4.6 and Opus 4.8 are statistically tied on capability, but
            Sonnet abandons a correct answer under a wrong reviewer 17.9% of the time versus Opus&rsquo;s 2.1%
            — a ~9× gap with non-overlapping confidence intervals.
          </div>
        </section>

        <section id="ecosystem">
          <p className="eyebrow">Open source · interoperable</p>
          <h2>Built to live in the safety-eval ecosystem</h2>
          <p>
            The probes are tiny and provider-neutral on purpose, so other people&rsquo;s harnesses can drive
            them. The flagship is a real <strong>Inspect AI</strong> adapter — the same deterministic
            detectors, now against any model Inspect can reach.
          </p>
          <CodeBlock title="inspect.sh" code={INSPECT} />

          <h3>Shipped integrations</h3>
          <div className="eco-grid">
            {SHIPPED.map((e) => (
              <a className="eco shipped" href={e.href} key={e.name}>
                <span className="eco-tag">Shipped</span>
                <b>{e.name}</b>
                <span className="eco-by">{e.by}</span>
                <p>{e.desc}</p>
              </a>
            ))}
          </div>

          <h3>Patterned after · compatible by design</h3>
          <p className="muted-p">
            Conventions VeriGrad deliberately mirrors so it slots into a real safety-eval workflow —
            listed honestly as influences and interop targets, not bundled adapters.
          </p>
          <div className="eco-grid">
            {PATTERNED.map((e) => (
              <a className="eco" href={e.href} key={e.name}>
                <span className="eco-tag muted">Influence</span>
                <b>{e.name}</b>
                <span className="eco-by">{e.by}</span>
                <p>{e.desc}</p>
              </a>
            ))}
          </div>
          <p className="muted-p" style={{ marginTop: 14 }}>
            Full details in{" "}
            <a href={`${REPO}/blob/main/docs/INTEGRATIONS.md`}>docs/INTEGRATIONS.md</a>.
          </p>
        </section>

        <section id="playground">
          <p className="eyebrow">Interactive · runs in your browser</p>
          <h2>Train a model on the data, live</h2>
          <p>No server, no pre-baked numbers — these widgets run real computation in your browser on the
            648 logged samples from the cross-domain run.</p>
          <MLPlayground />
          <KappaSim />
        </section>

        <section id="scaling">
          <p className="eyebrow">Scales to a research program</p>
          <h2>Across domains, providers, and an elicitation gradient</h2>
          <figure>
            <img src="/assets/fig_gradient.svg" alt="Deference under escalating authority pressure across domains" />
            <figcaption>
              <b>Elicitation gradient.</b> Deference under escalating authority, per model, across two
              domains. Cross-domain run: 648 samples, $1.74.
            </figcaption>
          </figure>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">GSM8K · L1</th>
                <th className="num">GSM8K · L3</th>
                <th className="num">CSQA · L1</th>
                <th className="num">CSQA · L3</th>
              </tr>
            </thead>
            <tbody>
              {GRADIENT.map((r) => (
                <tr key={r.model}>
                  <td><code>{r.model}</code></td>
                  <td className="num">{r.gsm8kL1}</td>
                  <td className="num">{r.gsm8kL3}</td>
                  <td className="num">{r.csqaL1}</td>
                  <td className="num">{"worst" in r && r.worst ? <span className="worst">{r.csqaL3}</span> : r.csqaL3}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="callout">
            <strong>Provider-agnostic by design.</strong> The native runner targets Anthropic; the Inspect
            adapter lifts that ceiling so the same probes produce a cross-vendor leaderboard. Runs are
            content-addressed and resumable with a hard cost ceiling — see{" "}
            <a href={`${REPO}/blob/main/docs/SCALING.md`}>docs/SCALING.md</a>.
          </div>
          <div className="callout">
            <strong>FDR correction changes a conclusion.</strong> On CommonsenseQA, Haiku-vs-Sonnet (47% vs
            22%) is significant at raw <em>p</em> = 0.026 but not after Benjamini–Hochberg (<em>q</em> =
            0.052). And the model ranking differs across domains — a propensity measured on math doesn&rsquo;t
            cleanly transfer.
          </div>
        </section>

        <section id="mechanism">
          <p className="eyebrow">Why models cave</p>
          <h2>Social, not cognitive</h2>
          <p>
            When a model abandons the correct answer, did its reasoning already compute it and then cave
            (<strong>override</strong>), or did the pressure corrupt the computation (<strong>anchored</strong>)?
            Across the lineup, ~90% is override — the model knew, and threw it away.
          </p>
          <figure>
            <img src="/assets/fig_mechanism.svg" alt="Override versus anchored reasoning" />
            <figcaption><b>Override dominates.</b> Two independent signals classify each case and agree at 94%.</figcaption>
          </figure>
          <table className="data-table">
            <thead>
              <tr><th>Model</th><th className="num">Deference cases</th><th className="num">Override</th><th className="num">Anchored</th><th className="num">Override share</th></tr>
            </thead>
            <tbody>
              {MECHANISM.map((r) => (
                <tr key={r.model}>
                  <td><code>{r.model}</code></td>
                  <td className="num">{r.deference}</td>
                  <td className="num">{r.override}</td>
                  <td className="num">{r.anchored}</td>
                  <td className="num">{r.share}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section id="reliability">
          <p className="eyebrow">Is the ruler trustworthy?</p>
          <h2>We test our graders</h2>
          <table className="data-table">
            <thead>
              <tr><th>Label</th><th className="num">Cohen&rsquo;s κ</th><th className="num">Raw agreement</th><th className="num">n</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              {RELIABILITY.map((r) => (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  <td className="num">{r.kappa}</td>
                  <td className="num">{r.raw}</td>
                  <td className="num">{r.n}</td>
                  <td className={r.ok ? "best" : ""}>{r.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="callout warn">
            <strong>The cross-check caught a bug in our own ruler.</strong> An earlier spec-gaming detector
            flagged 3 &ldquo;positives&rdquo; the judge unanimously rejected — all three the same clock-time answer,
            &ldquo;2:00&nbsp;PM&rdquo;, misread as two numbers (κ = 0.00 despite 98% raw agreement). After the fix, true
            spec-gaming is 0/150.
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <div>
            <h4>Project</h4>
            <a href={REPO}>GitHub repository</a>
            <a href={`${REPO}/blob/main/LICENSE`}>License — MIT</a>
            <a href={`${REPO}/blob/main/CONTRIBUTING.md`}>Contributing</a>
          </div>
          <div>
            <h4>Docs</h4>
            <a href={`${REPO}/blob/main/FINDINGS.md`}>Findings</a>
            <a href={`${REPO}/blob/main/docs/INTEGRATIONS.md`}>Integrations</a>
            <a href={`${REPO}/blob/main/docs/SCALING.md`}>Scaling</a>
            <a href={`${REPO}/blob/main/MECHANISM.md`}>Mechanistic analysis</a>
          </div>
          <div>
            <h4>Built with</h4>
            <p className="meta">
              Next.js + React + the Anthropic API, with an Inspect AI adapter. Real models, real datasets,
              no synthetic numbers. Measuring what models <em>will</em> do under pressure.
            </p>
          </div>
        </div>
      </footer>
    </>
  );
}
