"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { DocGroup } from "@/lib/docs";

export default function Sidebar({ groups }: { groups: DocGroup[] }) {
  const pathname = usePathname();
  return (
    <nav className="docs-side" aria-label="Documentation">
      {groups.map((g) => (
        <div className="docs-group" key={g.group}>
          <h4>{g.group}</h4>
          <ul>
            {g.items.map((it) => {
              const href = `/docs/${it.slug}`;
              const active = pathname === href;
              return (
                <li key={it.slug}>
                  <Link className={active ? "active" : ""} href={href}>
                    {it.title}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
