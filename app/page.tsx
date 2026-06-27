import LiveProbe from "@/components/LiveProbe";
import MLPlayground from "@/components/MLPlayground";
import KappaSim from "@/components/KappaSim";
import { LEADERBOARD, MECHANISM, GRADIENT, RELIABILITY, STATS } from "@/lib/data";

const REPO = "https://github.com/aravinds-kannappan/VeriGrad-RL";

export default function Home() {
  return (
    <>
      <header className="site-header">
        <div className="bar">
          <a className="brand" href="#top">
            <span className="dot" /> VeriGrad&nbsp;RL
          </a>
          <nav className="bar-links">
            <a href="#live">Live demo</a>
            <a href="#results">Results</a>
            <a href="#playground">Playground</a>
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
            <span className="tag">Model Evaluation</span>
            <span className="tag">Next.js · Live</span>
          </div>
          <h1>Watch a frontier model cave under pressure — live.</h1>
          <p className="tagline">
            VeriGrad RL measures the <strong>propensities</strong> of real frontier models — sycophancy,
            spec-gaming, reasoning faithfulness. This isn&rsquo;t a static page: probe a live model below,
            train a classifier in your browser, and explore the real run data.
          </p>
          <div className="badges">
            <img src={`https://img.shields.io/github/actions/workflow/status/aravinds-kannappan/VeriGrad-RL/ci.yml?branch=main&label=CI`} alt="CI" />
            <img src="https://img.shields.io/badge/license-MIT-0f766e" alt="MIT" />
            <img src="https://img.shields.io/badge/Next.js-app-000000" alt="Next.js" />
            <img src="https://img.shields.io/badge/tests-49%20passing-16a34a" alt="tests" />
            <img src={`https://img.shields.io/github/stars/aravinds-kannappan/VeriGrad-RL?style=social`} alt="stars" />
          </div>
          <div className="cta">
            <a className="btn primary" href="#live">Try the live demo ↓</a>
            <a className="btn" href={REPO}>View on GitHub</a>
            <a className="btn" href={`${REPO}/blob/main/FINDINGS.md`}>Read the findings</a>
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
          <h2>Across domains, under an elicitation gradient</h2>
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
            <a href={`${REPO}/blob/main/MECHANISM.md`}>Mechanistic analysis</a>
            <a href={`${REPO}/blob/main/benchmark/scale/REPORT.md`}>Scaling report</a>
          </div>
          <div>
            <h4>Built with</h4>
            <p className="meta">
              Next.js + React + the Anthropic API. Real models, real datasets, no synthetic numbers.
              Measuring what models <em>will</em> do under pressure.
            </p>
          </div>
        </div>
      </footer>
    </>
  );
}
