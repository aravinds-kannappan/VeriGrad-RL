import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { ALL_DOCS, getDoc, renderDoc, REPO } from "@/lib/docs";

export const dynamicParams = false;

export function generateStaticParams() {
  return ALL_DOCS.map((d) => ({ slug: d.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const item = getDoc(slug);
  return { title: item ? `${item.title} · VeriGrad RL docs` : "VeriGrad RL docs" };
}

export default async function DocPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getDoc(slug);
  if (!item) notFound();
  const { html } = renderDoc(item);
  return (
    <article className="doc-content">
      <div className="doc-meta">
        <Link href="/docs/getting-started">Docs</Link>
        <span>/</span>
        <span>{item.title}</span>
        <a className="doc-edit" href={`${REPO}/blob/main/${item.file}`}>
          Edit on GitHub →
        </a>
      </div>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </article>
  );
}
