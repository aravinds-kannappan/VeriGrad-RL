import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  themeColor: "#0b1f33",
};

export const metadata: Metadata = {
  title: "VeriGrad RL — propensity evaluation for frontier models",
  description:
    "An interactive toolkit measuring what frontier models do under pressure — sycophancy, spec-gaming, reasoning faithfulness — with a live probe, in-browser ML, clustered CIs, and FDR correction.",
  icons: {
    icon:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230b1f33'/%3E%3Ccircle cx='16' cy='16' r='6' fill='%230f766e'/%3E%3C/svg%3E",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
