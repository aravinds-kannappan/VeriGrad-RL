# Biosafety Playground

The biosafety playground applies VeriGrad RL to a real-world inspired safety workflow: synthetic DNA order-screening and dual-use request triage.

It is intentionally defensive and non-operational:

- no real pathogen sequences,
- no wet-lab protocols,
- no biological optimization instructions,
- no sequence-design assistance.

Instead, it uses non-sensitive mock screening fingerprints and synthetic risk features:

- `sequence_risk`
- `capability_risk`
- `synthesis_scale`
- `benign_research_value`
- `uncertainty`
- `review_pressure`

The browser playground lets users adjust these values and watch:

- recommended intervention,
- behavior logits,
- verifier reward,
- safety/utility/escalation/mechanistic scores,
- causal attribution,
- intervention comparison.

## Local File

Open:

```text
docs/biosafety.html
```

## Vercel

The repo includes `vercel.json`. Production deploys run `node scripts/build_site.mjs`, which
copies the static `docs/` site into Vercel's `public/` output directory.

Expected routes:

- `/` for the project site,
- `/biosafety` for the interactive playground,
- `/assets/*` for generated figures.

To deploy through Vercel:

1. Import `aravinds-kannappan/VeriGrad-RL`.
2. Set Framework Preset to `Other`.
3. Keep Root Directory empty.
4. Let `vercel.json` control the build command and output directory.
