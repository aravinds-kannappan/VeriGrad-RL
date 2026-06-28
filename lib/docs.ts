// Server-only docs loader: reads the repo's Markdown files at build time and
// renders them to HTML, rewriting repo-relative links to the in-site docs (or to
// GitHub for things that don't live in the docs site). Used by /docs.

import fs from "node:fs";
import path from "node:path";
import { marked } from "marked";

export const REPO = "https://github.com/aravinds-kannappan/VeriGrad-RL";
const RAW = "https://raw.githubusercontent.com/aravinds-kannappan/VeriGrad-RL/main";

export type DocItem = { slug: string; title: string; file: string; blurb: string };
export type DocGroup = { group: string; items: DocItem[] };

export const DOC_GROUPS: DocGroup[] = [
  {
    group: "Get started",
    items: [
      { slug: "getting-started", title: "Getting started", file: "docs/GETTING_STARTED.md", blurb: "Install, run the benchmark and the RL baseline." },
      { slug: "architecture", title: "Architecture", file: "docs/ARCHITECTURE.md", blurb: "How the pieces fit together." },
    ],
  },
  {
    group: "Findings",
    items: [
      { slug: "findings", title: "Findings", file: "FINDINGS.md", blurb: "The core capability-vs-propensity result." },
      { slug: "mechanism", title: "Mechanistic analysis", file: "MECHANISM.md", blurb: "Override vs. anchored: why models cave." },
    ],
  },
  {
    group: "Systems",
    items: [
      { slug: "circuit-discovery", title: "Circuit discovery", file: "docs/MECH_INTERP.md", blurb: "ACDC + path patching on a transparent safety circuit." },
      { slug: "integrations", title: "Integrations", file: "docs/INTEGRATIONS.md", blurb: "Inspect AI adapter + interoperability." },
      { slug: "scaling", title: "Scaling", file: "docs/SCALING.md", blurb: "Breadth, rigor, and platform for a research program." },
    ],
  },
  {
    group: "Reference",
    items: [
      { slug: "api", title: "API reference", file: "docs/API_REFERENCE.md", blurb: "The Python package surface." },
      { slug: "references", title: "Papers & references", file: "docs/REFERENCES.md", blurb: "The AI-safety research behind VeriGrad." },
    ],
  },
];

export const ALL_DOCS: DocItem[] = DOC_GROUPS.flatMap((g) => g.items);

// basename ("INTEGRATIONS.md") -> in-site slug, for rewriting cross-doc links.
const SLUG_BY_BASENAME = new Map(ALL_DOCS.map((d) => [path.basename(d.file), d.slug]));

export function getDoc(slug: string): DocItem | undefined {
  return ALL_DOCS.find((d) => d.slug === slug);
}

function rewriteLinks(html: string): string {
  // <a href="..."> — keep http/anchors; map .md to in-site docs; else point at GitHub.
  html = html.replace(/href="([^"]+)"/g, (_m, href: string) => {
    if (/^(https?:|#|mailto:)/.test(href)) return `href="${href}"`;
    const clean = href.replace(/^\.?\/?/, "").replace(/^(\.\.\/)+/, "");
    if (/\.md(#.*)?$/i.test(clean)) {
      const [pathPart, hash] = clean.split("#");
      const slug = SLUG_BY_BASENAME.get(path.basename(pathPart));
      if (slug) return `href="/docs/${slug}${hash ? "#" + hash : ""}"`;
    }
    return `href="${REPO}/blob/main/${clean}"`;
  });
  // <img src="..."> — relative images resolve from the raw GitHub tree.
  html = html.replace(/src="([^"]+)"/g, (_m, src: string) => {
    if (/^(https?:|data:|\/)/.test(src)) return `src="${src}"`;
    return `src="${RAW}/${src.replace(/^\.?\/?/, "")}"`;
  });
  return html;
}

export function renderDoc(item: DocItem): { title: string; html: string } {
  const raw = fs.readFileSync(path.join(process.cwd(), item.file), "utf-8");
  const html = rewriteLinks(marked.parse(raw, { gfm: true, breaks: false }) as string);
  return { title: item.title, html };
}
