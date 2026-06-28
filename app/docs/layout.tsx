import Link from "next/link";
import Sidebar from "./Sidebar";
import { DOC_GROUPS, REPO } from "@/lib/docs";

export default function DocsLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <header className="site-header">
        <div className="bar">
          <Link className="brand" href="/">
            <span className="dot" /> VeriGrad&nbsp;RL
          </Link>
          <nav className="bar-links">
            <Link href="/">Home</Link>
            <Link href="/docs/getting-started">Docs</Link>
            <a className="ghstar" href={REPO}>★ GitHub</a>
          </nav>
        </div>
      </header>
      <div className="docs-shell">
        <Sidebar groups={DOC_GROUPS} />
        <main className="docs-main">{children}</main>
      </div>
    </>
  );
}
