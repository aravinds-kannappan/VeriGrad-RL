# VeriGrad RL — interactive web app

A real **Next.js 15 (App Router) + React + TypeScript** application — not a static page. It
has a **live API route** (`app/api/probe/route.ts`) that calls real frontier models in real
time, plus in-browser machine learning (logistic regression trained client-side) and the
κ-paradox simulator.

## Run locally

```bash
cd web
npm install
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env.local   # enables the live probe
npm run dev        # http://localhost:3000
# or:
npm run build && npm start
```

The site fully works **without** a key — only the live "probe a model" demo needs one (it
shows a friendly "disabled" message otherwise).

## Deploy on Vercel (two one-time settings)

This app lives in the `web/` subdirectory of the repo, so point Vercel at it:

1. **Project → Settings → Build & Deployment → Root Directory →** set to `web`.
   Vercel then auto-detects Next.js (no `vercel.json` needed).
2. **Project → Settings → Environment Variables →** add `ANTHROPIC_API_KEY`
   (used server-side by the `/api/probe` route; never exposed to the browser).

Then every push to `main` auto-deploys.

> ⚠️ The `/api/probe` route calls a paid model with your key. `max_tokens` is capped and the
> input length is bounded, but it is a public endpoint once deployed — add rate limiting (or
> leave the key unset to keep it demo-disabled) if abuse is a concern.

## Regenerate the embedded data

The app's data comes from the real runs:

```bash
python scripts/build_web_data.py   # -> web/lib/samples.json, web/lib/problems.json, web/public/assets/*.svg
```

## Structure

```
web/
  app/
    page.tsx            Homepage composition (server component)
    layout.tsx          Metadata + globals
    globals.css         Design system
    api/probe/route.ts  Live model call (server-side, real time)
  components/
    LiveProbe.tsx       Interactive live demo (client)
    MLPlayground.tsx    In-browser logistic regression (client)
    KappaSim.tsx        κ-paradox simulator (client)
  lib/
    data.ts             Real leaderboard / mechanism / gradient numbers
    samples.json        648 real samples for the in-browser ML
    problems.json       Real GSM8K problems for the live probe
```
