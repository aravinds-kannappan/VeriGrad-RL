// Real results from the logged runs. Numbers mirror benchmark/results/summary.json
// and benchmark/scale/summary.json — nothing synthetic.

export type Row = {
  model: string;
  control: string;
  deference: string;
  sycophancy: string;
  sycophancyCI: string;
  spec: string;
  cost: string;
  tone?: "best" | "worst";
};

export const LEADERBOARD: Row[] = [
  { model: "opus-4.8", control: "96.7%", deference: "2.7%", sycophancy: "2.1%", sycophancyCI: "[0.7, 5.9]", spec: "0.0%", cost: "$1.78", tone: "best" },
  { model: "sonnet-4.6", control: "96.7%", deference: "20.0%", sycophancy: "17.9%", sycophancyCI: "[12.5, 25.0]", spec: "0.0%", cost: "$1.56", tone: "worst" },
  { model: "haiku-4.5", control: "95.3%", deference: "10.7%", sycophancy: "9.1%", sycophancyCI: "[5.4, 14.9]", spec: "0.0%", cost: "$0.66" },
];

export const MECHANISM = [
  { model: "opus-4.8", deference: 4, override: 3, anchored: 1, share: "75.0%" },
  { model: "sonnet-4.6", deference: 30, override: 28, anchored: 2, share: "93.3%" },
  { model: "haiku-4.5", deference: 16, override: 14, anchored: 2, share: "87.5%" },
];

// Cross-domain deference (%) at authority intensity L1 and L3.
export const GRADIENT = [
  { model: "opus-4.8", gsm8kL1: "0.0%", gsm8kL3: "13.9%", csqaL1: "2.8%", csqaL3: "2.8%" },
  { model: "sonnet-4.6", gsm8kL1: "0.0%", gsm8kL3: "8.3%", csqaL1: "8.3%", csqaL3: "22.2%" },
  { model: "haiku-4.5", gsm8kL1: "0.0%", gsm8kL3: "16.7%", csqaL1: "8.3%", csqaL3: "47.2%", worst: true },
];

export const RELIABILITY = [
  { label: "Correctness", kappa: "0.95", raw: "99%", n: "450", verdict: "validated", ok: true },
  { label: "Deference", kappa: "0.97", raw: "99%", n: "150", verdict: "validated", ok: true },
  { label: "Spec-gaming", kappa: "—", raw: "—", n: "150", verdict: "0 positives — nothing to validate", ok: false },
];

export const MODELS = [
  { key: "opus", label: "Opus 4.8" },
  { key: "sonnet", label: "Sonnet 4.6" },
  { key: "haiku", label: "Haiku 4.5" },
];

export const STATS = [
  { v: "9×", l: "Sycophancy spread" },
  { v: "2", l: "Task domains" },
  { v: "κ 0.95", l: "Grader validated" },
  { v: "0", l: "Errors across runs" },
];
