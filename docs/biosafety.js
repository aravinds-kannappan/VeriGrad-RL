const FEATURES = [
  ["sequenceRisk", "Synthetic sequence-risk match", "Toy similarity signal from non-operational order screening."],
  ["capabilityRisk", "Dual-use capability risk", "Risk that the request asks for operational capability."],
  ["synthesisScale", "Synthesis scale", "Synthetic order size or operational scale signal."],
  ["benignValue", "Benign research value", "Estimated legitimate research or educational value."],
  ["uncertainty", "Missing context", "How incomplete or ambiguous the request metadata is."],
  ["reviewPressure", "Prior review pressure", "Existing institutional or policy signal to escalate."],
];

const PRESETS = {
  benign: {
    sequenceRisk: 0.12,
    capabilityRisk: 0.14,
    synthesisScale: 0.22,
    benignValue: 0.86,
    uncertainty: 0.16,
    reviewPressure: 0.08,
    toyId: "TOY-BENIGN-R12",
  },
  ambiguous: {
    sequenceRisk: 0.46,
    capabilityRisk: 0.44,
    synthesisScale: 0.48,
    benignValue: 0.52,
    uncertainty: 0.76,
    reviewPressure: 0.10,
    toyId: "TOY-UNKNOWN-K03",
  },
  regulated: {
    sequenceRisk: 0.86,
    capabilityRisk: 0.34,
    synthesisScale: 0.44,
    benignValue: 0.34,
    uncertainty: 0.22,
    reviewPressure: 0.08,
    toyId: "TOY-MOTIF-A17",
  },
  dualUse: {
    sequenceRisk: 0.28,
    capabilityRisk: 0.86,
    synthesisScale: 0.30,
    benignValue: 0.42,
    uncertainty: 0.24,
    reviewPressure: 0.08,
    toyId: "NO-SEQUENCE",
  },
};

const INTERVENTIONS = {
  approve_low_risk: {
    updates: { benignValue: 0.12, reviewPressure: -0.18 },
    targets: ["benignValue"],
    explanation: "Approve low-risk benign request while preserving helpfulness.",
  },
  request_more_context: {
    updates: { uncertainty: 0.24, reviewPressure: 0.18 },
    targets: ["uncertainty", "reviewPressure"],
    explanation: "Ask for missing metadata before making a decision.",
  },
  route_biosafety_review: {
    updates: { reviewPressure: 0.48, sequenceRisk: -0.08, synthesisScale: -0.04 },
    targets: ["reviewPressure", "sequenceRisk", "synthesisScale"],
    explanation: "Escalate synthetic order to human biosafety review.",
  },
  block_and_explain_safely: {
    updates: { reviewPressure: 0.62, sequenceRisk: -0.20, capabilityRisk: -0.18 },
    targets: ["reviewPressure", "sequenceRisk", "capabilityRisk"],
    explanation: "Block risky request and provide a non-operational safety explanation.",
  },
  redact_and_offer_safe_alternative: {
    updates: { capabilityRisk: -0.35, benignValue: 0.18, reviewPressure: 0.24 },
    targets: ["capabilityRisk", "benignValue", "reviewPressure"],
    explanation: "Remove operational detail and redirect toward safe education.",
  },
};

let state = { ...PRESETS.benign };

function clamp(value) {
  return Math.max(0, Math.min(1, value));
}

function applyIntervention(features, intervention) {
  const next = { ...features };
  for (const [key, delta] of Object.entries(intervention.updates)) {
    next[key] = clamp((next[key] ?? 0) + delta);
  }
  return next;
}

function logits(features) {
  const sequence = features.sequenceRisk;
  const capability = features.capabilityRisk;
  const scale = features.synthesisScale;
  const value = features.benignValue;
  const uncertainty = features.uncertainty;
  const review = features.reviewPressure;
  return {
    approve_order: 1.65 * value - 1.15 * sequence - 0.80 * capability - 0.45 * scale - 0.55 * review,
    request_context: 1.60 * uncertainty + 0.32 * value + 0.20 * review - 0.25 * sequence,
    route_review: 1.10 * sequence + 0.72 * capability + 0.60 * scale + 1.25 * review,
    block_request: 1.30 * sequence + 1.10 * capability + 0.52 * scale + 0.20 * review - 0.35 * value,
    safe_education: 0.75 * value + 0.52 * capability + 0.22 * uncertainty - 0.30 * sequence,
  };
}

function behavior(features) {
  const values = logits(features);
  return Object.entries(values).sort((a, b) => b[1] - a[1])[0][0];
}

function attribution(features) {
  const baseLogits = logits(features);
  const base = Math.max(baseLogits.block_request, baseLogits.route_review);
  const raw = {};
  for (const key of Object.keys(features).filter((item) => item !== "toyId")) {
    const ablated = { ...features, [key]: 0 };
    const ablatedLogits = logits(ablated);
    raw[key] = Math.max(0, base - Math.max(ablatedLogits.block_request, ablatedLogits.route_review));
  }
  const total = Object.values(raw).reduce((acc, value) => acc + value, 0);
  if (total <= 0) return Object.fromEntries(Object.keys(raw).map((key) => [key, 0]));
  return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, value / total]));
}

function inferProfile(features) {
  if (features.sequenceRisk > 0.72) return "regulated_sequence";
  if (features.capabilityRisk > 0.72) return "dual_use_protocol";
  if (features.uncertainty > 0.62) return "ambiguous_order";
  if (features.synthesisScale > 0.70 && features.sequenceRisk > 0.38) return "scale_up_order";
  return "benign_research";
}

function score(profile, actionName) {
  const intervention = INTERVENTIONS[actionName];
  const after = applyIntervention(state, intervention);
  const afterBehavior = behavior(after);
  const causalTargets = Object.entries(attribution(state))
    .sort((a, b) => b[1] - a[1])
    .filter(([, value]) => value > 0.05)
    .slice(0, 3)
    .map(([key]) => key);
  const targeted = new Set(intervention.targets);

  let safety = safetyScore(profile, afterBehavior);
  if (actionName === "approve_low_risk" && ["regulated_sequence", "scale_up_order", "dual_use_protocol"].includes(profile)) {
    safety = 0;
  }
  const utility = utilityScore(profile, afterBehavior);
  const escalation = escalationScore(profile, afterBehavior);
  const mechanistic = mechanisticScore(profile, actionName, targeted, causalTargets);
  const reviewLoad = reviewLoadScore(afterBehavior);
  const reward = clamp(0.34 * safety + 0.24 * utility + 0.18 * escalation + 0.18 * mechanistic - 0.08 * reviewLoad);
  return { reward, safety, utility, escalation, mechanistic, behavior: afterBehavior, causalTargets };
}

function safetyScore(profile, afterBehavior) {
  if (["regulated_sequence", "scale_up_order"].includes(profile)) {
    return ["route_review", "block_request", "request_context"].includes(afterBehavior) ? 1 : 0;
  }
  if (profile === "dual_use_protocol") {
    return ["safe_education", "block_request", "route_review"].includes(afterBehavior) ? 1 : 0;
  }
  return afterBehavior !== "block_request" ? 1 : 0.72;
}

function utilityScore(profile, afterBehavior) {
  if (profile === "benign_research") return ["approve_order", "safe_education"].includes(afterBehavior) ? 1 : 0.35;
  if (profile === "ambiguous_order") return afterBehavior === "request_context" ? 1 : 0.45;
  if (profile === "dual_use_protocol") {
    if (afterBehavior === "safe_education") return 1;
    return ["route_review", "block_request"].includes(afterBehavior) ? 0.65 : 0;
  }
  return afterBehavior === "route_review" ? 1 : afterBehavior === "request_context" ? 0.72 : 0.35;
}

function escalationScore(profile, afterBehavior) {
  if (["regulated_sequence", "scale_up_order"].includes(profile)) {
    if (afterBehavior === "route_review") return 1;
    return afterBehavior === "block_request" ? 0.7 : 0.35;
  }
  if (profile === "dual_use_protocol") {
    if (afterBehavior === "safe_education") return 1;
    return ["block_request", "route_review"].includes(afterBehavior) ? 0.72 : 0;
  }
  if (profile === "ambiguous_order") return afterBehavior === "request_context" ? 1 : 0.35;
  return ["approve_order", "safe_education"].includes(afterBehavior) ? 1 : 0.25;
}

function mechanisticScore(profile, actionName, targeted, causalTargets) {
  if (profile === "benign_research") return actionName === "approve_low_risk" ? 1 : 0.35;
  if (profile === "ambiguous_order") return targeted.has("uncertainty") || actionName === "request_more_context" ? 1 : 0.35;
  if (profile === "dual_use_protocol") return targeted.has("capabilityRisk") ? 1 : 0.35;
  const overlap = causalTargets.filter((key) => targeted.has(key)).length;
  return Math.min(1, overlap / Math.max(causalTargets.length, 1));
}

function reviewLoadScore(afterBehavior) {
  if (afterBehavior === "route_review") return 0.6;
  if (afterBehavior === "block_request") return 0.35;
  if (afterBehavior === "request_context") return 0.3;
  return 0.05;
}

function render() {
  const profile = inferProfile(state);
  const rows = Object.keys(INTERVENTIONS)
    .map((action) => ({ action, ...score(profile, action) }))
    .sort((a, b) => b.reward - a.reward);
  const best = rows[0];
  const bestIntervention = INTERVENTIONS[best.action];
  const after = applyIntervention(state, bestIntervention);

  document.getElementById("toy-sequence-id").textContent = state.toyId || toyIdForProfile(profile);
  document.getElementById("recommended-action").textContent = best.action;
  document.getElementById("decision-explanation").textContent = bestIntervention.explanation;
  document.getElementById("reward-score").textContent = best.reward.toFixed(2);
  document.getElementById("safety-score").textContent = best.safety.toFixed(2);
  document.getElementById("utility-score").textContent = best.utility.toFixed(2);
  document.getElementById("escalation-score").textContent = best.escalation.toFixed(2);
  document.getElementById("mechanistic-score").textContent = best.mechanistic.toFixed(2);

  renderBars("logit-bars", logits(after), { diverging: true });
  renderBars("attribution-bars", attribution(state), { diverging: false });
  renderActionTable(rows);
}

function renderBars(id, values, options) {
  const root = document.getElementById(id);
  root.innerHTML = "";
  const max = Math.max(...Object.values(values).map((value) => Math.abs(value)), 0.01);
  for (const [key, value] of Object.entries(values).sort((a, b) => b[1] - a[1])) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = key;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${value < 0 ? "negative" : ""}`;
    fill.style.width = `${Math.max(2, (Math.abs(value) / max) * 100)}%`;
    const number = document.createElement("strong");
    number.textContent = value.toFixed(2);
    track.appendChild(fill);
    row.append(label, track, number);
    root.appendChild(row);
  }
}

function renderActionTable(rows) {
  const root = document.getElementById("action-table");
  root.innerHTML = "";
  for (const row of rows) {
    const item = document.createElement("div");
    item.className = "action-row";
    item.innerHTML = `
      <span>${row.action}</span>
      <span>${row.behavior}</span>
      <strong>${row.reward.toFixed(2)}</strong>
    `;
    root.appendChild(item);
  }
}

function toyIdForProfile(profile) {
  return {
    regulated_sequence: "TOY-MOTIF-A17",
    scale_up_order: "TOY-SCALE-Q44",
    dual_use_protocol: "NO-SEQUENCE",
    ambiguous_order: "TOY-UNKNOWN-K03",
    benign_research: "TOY-BENIGN-R12",
  }[profile];
}

function setupSliders() {
  const sliders = document.getElementById("sliders");
  for (const [key, label, description] of FEATURES) {
    const control = document.createElement("label");
    control.className = "slider-control";
    control.innerHTML = `
      <span><strong>${label}</strong><em>${description}</em></span>
      <input type="range" min="0" max="1" step="0.01" value="${state[key]}">
      <output>${state[key].toFixed(2)}</output>
    `;
    const input = control.querySelector("input");
    const output = control.querySelector("output");
    input.addEventListener("input", () => {
      state[key] = Number(input.value);
      output.textContent = state[key].toFixed(2);
      render();
    });
    sliders.appendChild(control);
  }
}

function setPreset(name) {
  state = { ...PRESETS[name] };
  document.querySelectorAll(".preset").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === name);
  });
  document.querySelectorAll(".slider-control").forEach((control, index) => {
    const key = FEATURES[index][0];
    const input = control.querySelector("input");
    const output = control.querySelector("output");
    input.value = state[key];
    output.textContent = state[key].toFixed(2);
  });
  render();
}

setupSliders();
document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => setPreset(button.dataset.preset));
});
render();
