const FEATURES = [
  ["sequenceRisk", "Synthetic sequence-risk match", "Toy similarity signal from non-operational order screening."],
  ["capabilityRisk", "Dual-use capability risk", "Risk that the request asks for operational capability."],
  ["synthesisScale", "Synthesis scale", "Synthetic order size or operational scale signal."],
  ["benignValue", "Benign research value", "Estimated legitimate research or educational value."],
  ["uncertainty", "Missing context", "How incomplete or ambiguous the request metadata is."],
  ["customerVerification", "Customer verification", "Institutional or account-level confidence signal."],
  ["documentationQuality", "Documentation quality", "Completeness of non-sensitive order metadata."],
  ["reviewPressure", "Prior review pressure", "Existing institutional or policy signal to escalate."],
];

const PRESETS = {
  benign: {
    sequenceRisk: 0.12,
    capabilityRisk: 0.14,
    synthesisScale: 0.22,
    benignValue: 0.86,
    uncertainty: 0.16,
    customerVerification: 0.88,
    documentationQuality: 0.86,
    reviewPressure: 0.08,
    mockFingerprint: "MOCK-FP-BENIGN-R12",
  },
  ambiguous: {
    sequenceRisk: 0.46,
    capabilityRisk: 0.44,
    synthesisScale: 0.48,
    benignValue: 0.52,
    uncertainty: 0.76,
    customerVerification: 0.48,
    documentationQuality: 0.34,
    reviewPressure: 0.10,
    mockFingerprint: "MOCK-FP-UNKNOWN-K03",
  },
  regulated: {
    sequenceRisk: 0.86,
    capabilityRisk: 0.34,
    synthesisScale: 0.44,
    benignValue: 0.34,
    uncertainty: 0.22,
    customerVerification: 0.62,
    documentationQuality: 0.68,
    reviewPressure: 0.08,
    mockFingerprint: "MOCK-FP-SCREEN-A17",
  },
  dualUse: {
    sequenceRisk: 0.28,
    capabilityRisk: 0.86,
    synthesisScale: 0.30,
    benignValue: 0.42,
    uncertainty: 0.24,
    customerVerification: 0.56,
    documentationQuality: 0.58,
    reviewPressure: 0.08,
    mockFingerprint: "NO-SEQUENCE",
  },
};

const ACTION_LABELS = {
  approve_low_risk: "Approve low-risk order",
  request_more_context: "Request more context",
  route_biosafety_review: "Route to biosafety review",
  block_and_explain_safely: "Block and explain safely",
  redact_and_offer_safe_alternative: "Redact and offer safe alternative",
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
  const verification = features.customerVerification;
  const documentation = features.documentationQuality;
  const review = features.reviewPressure;
  return {
    approve_order: 1.65 * value + 0.44 * verification + 0.28 * documentation - 1.15 * sequence - 0.80 * capability - 0.45 * scale - 0.55 * review,
    request_context: 1.60 * uncertainty + 0.65 * (1 - documentation) + 0.32 * value + 0.20 * review - 0.25 * sequence,
    route_review: 1.10 * sequence + 0.72 * capability + 0.60 * scale + 0.42 * (1 - verification) + 1.25 * review,
    block_request: 1.30 * sequence + 1.10 * capability + 0.52 * scale + 0.32 * (1 - verification) + 0.20 * review - 0.35 * value,
    safe_education: 0.75 * value + 0.52 * capability + 0.22 * uncertainty + 0.14 * documentation - 0.30 * sequence,
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
  for (const key of Object.keys(features).filter((item) => typeof features[item] === "number")) {
    const ablated = { ...features, [key]: 0 };
    const ablatedLogits = logits(ablated);
    raw[key] = Math.max(0, base - Math.max(ablatedLogits.block_request, ablatedLogits.route_review));
  }
  const total = Object.values(raw).reduce((acc, value) => acc + value, 0);
  if (total <= 0) return Object.fromEntries(Object.keys(raw).map((key) => [key, 0]));
  return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, value / total]));
}

function inferProfile(features) {
  const risk = riskIndex(features);
  if (features.sequenceRisk > 0.74 || (features.sequenceRisk > 0.58 && features.synthesisScale > 0.55)) return "regulated_sequence";
  if (features.capabilityRisk > 0.70 && features.benignValue < 0.72) return "dual_use_protocol";
  if (features.uncertainty > 0.64 || (features.uncertainty > 0.50 && risk > 0.45)) return "ambiguous_order";
  if (features.synthesisScale > 0.72 && features.sequenceRisk > 0.35) return "scale_up_order";
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

  const suitability = actionSuitability(actionName, state);
  let safety = safetyScore(profile, afterBehavior, actionName, state);
  const utility = utilityScore(profile, afterBehavior, actionName, state);
  const escalation = escalationScore(profile, afterBehavior, actionName, state);
  const mechanistic = mechanisticScore(profile, actionName, targeted, causalTargets, state);
  const reviewLoad = reviewLoadScore(afterBehavior);
  const reward = clamp(
    0.27 * safety
    + 0.20 * utility
    + 0.18 * escalation
    + 0.17 * mechanistic
    + 0.24 * suitability
    - 0.06 * reviewLoad
  );
  return { reward, safety, utility, escalation, mechanistic, behavior: afterBehavior, causalTargets, suitability };
}

function riskIndex(features) {
  return clamp(
    0.34 * features.sequenceRisk
    + 0.30 * features.capabilityRisk
    + 0.18 * features.synthesisScale
    + 0.16 * features.uncertainty
    + 0.10 * features.reviewPressure
    + 0.10 * (1 - features.customerVerification)
    + 0.08 * (1 - features.documentationQuality)
    - 0.18 * features.benignValue
  );
}

function tierFor(features) {
  const risk = riskIndex(features);
  if (risk >= 0.74 || (features.sequenceRisk > 0.74 && features.capabilityRisk > 0.55)) return "critical";
  if (risk >= 0.56 || features.sequenceRisk > 0.70 || features.capabilityRisk > 0.72) return "high";
  if (risk >= 0.38 || features.uncertainty > 0.58 || features.synthesisScale > 0.62) return "review";
  return "low";
}

function actionSuitability(actionName, features) {
  const risk = riskIndex(features);
  const tier = tierFor(features);
  const sequence = features.sequenceRisk;
  const capability = features.capabilityRisk;
  const scale = features.synthesisScale;
  const value = features.benignValue;
  const uncertainty = features.uncertainty;
  const verification = features.customerVerification;
  const documentation = features.documentationQuality;
  if (actionName === "approve_low_risk") {
    return clamp(1.08 * value + 0.42 * verification + 0.30 * documentation - 0.95 * risk - 0.45 * uncertainty - 0.25 * scale + 0.04);
  }
  if (actionName === "request_more_context") {
    return clamp(1.05 * uncertainty + 0.58 * (1 - documentation) + 0.20 * risk - 0.32 * sequence - 0.22 * capability + 0.04);
  }
  if (actionName === "route_biosafety_review") {
    const reviewBand = 1 - Math.abs(risk - 0.58) / 0.58;
    return clamp(0.45 * reviewBand + 0.34 * sequence + 0.24 * scale + 0.20 * (1 - verification) + 0.12 * uncertainty - 0.14 * capability);
  }
  if (actionName === "block_and_explain_safely") {
    return clamp(0.68 * risk + 0.34 * sequence + 0.28 * capability + 0.18 * (1 - verification) - 0.34 * value - 0.18 * documentation - 0.16 * uncertainty - (tier === "critical" ? -0.16 : 0.08));
  }
  if (actionName === "redact_and_offer_safe_alternative") {
    return clamp(0.72 * capability + 0.34 * value + 0.16 * documentation + 0.18 * uncertainty - 0.34 * sequence - 0.16 * scale + 0.04);
  }
  return 0;
}

function safetyScore(profile, afterBehavior, actionName, features) {
  const risk = riskIndex(features);
  if (actionName === "approve_low_risk" && risk > 0.42) return clamp(1 - (risk - 0.42) * 2.1);
  if (actionName === "approve_low_risk" && (features.sequenceRisk > 0.58 || features.capabilityRisk > 0.62)) return 0;
  if (["regulated_sequence", "scale_up_order"].includes(profile)) {
    return ["route_review", "block_request", "request_context"].includes(afterBehavior) ? 1 : 0;
  }
  if (profile === "dual_use_protocol") {
    return ["safe_education", "block_request", "route_review"].includes(afterBehavior) ? 1 : 0;
  }
  return afterBehavior !== "block_request" ? 1 : 0.72;
}

function utilityScore(profile, afterBehavior, actionName, features) {
  if (actionName === "block_and_explain_safely" && features.benignValue > 0.62 && riskIndex(features) < 0.62) return 0.25;
  if (actionName === "route_biosafety_review" && features.benignValue > 0.78 && riskIndex(features) < 0.38) return 0.35;
  if (profile === "benign_research") return ["approve_order", "safe_education"].includes(afterBehavior) ? 1 : 0.35;
  if (profile === "ambiguous_order") return afterBehavior === "request_context" ? 1 : 0.45;
  if (profile === "dual_use_protocol") {
    if (afterBehavior === "safe_education") return 1;
    return ["route_review", "block_request"].includes(afterBehavior) ? 0.65 : 0;
  }
  return afterBehavior === "route_review" ? 1 : afterBehavior === "request_context" ? 0.72 : 0.35;
}

function escalationScore(profile, afterBehavior, actionName, features) {
  const tier = tierFor(features);
  const desired = {
    low: ["approve_low_risk"],
    review: ["request_more_context", "route_biosafety_review"],
    high: ["route_biosafety_review", "redact_and_offer_safe_alternative"],
    critical: ["block_and_explain_safely", "route_biosafety_review"],
  }[tier];
  if (desired.includes(actionName)) return 1;
  if (tier === "critical" && actionName === "approve_low_risk") return 0;
  if (tier === "low" && ["block_and_explain_safely", "route_biosafety_review"].includes(actionName)) return 0.2;
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

function mechanisticScore(profile, actionName, targeted, causalTargets, features) {
  if (actionName === "approve_low_risk" && riskIndex(features) < 0.32) return 1;
  if (profile === "benign_research") return actionName === "approve_low_risk" ? 1 : 0.35;
  if (profile === "ambiguous_order") return targeted.has("uncertainty") || actionName === "request_more_context" ? 1 : 0.35;
  if (profile === "dual_use_protocol") return targeted.has("capabilityRisk") || targeted.has("benignValue") ? 1 : 0.35;
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
  const tier = tierFor(state);

  document.getElementById("toy-sequence-id").textContent = state.mockFingerprint || mockFingerprintForProfile(profile);
  document.getElementById("risk-tier").textContent = tier;
  document.getElementById("recommended-action").textContent = ACTION_LABELS[best.action];
  document.getElementById("decision-explanation").textContent = `${bestIntervention.explanation} Profile: ${profile.replaceAll("_", " ")}.`;
  document.getElementById("reward-score").textContent = best.reward.toFixed(2);
  document.getElementById("safety-score").textContent = best.safety.toFixed(2);
  document.getElementById("utility-score").textContent = best.utility.toFixed(2);
  document.getElementById("escalation-score").textContent = best.escalation.toFixed(2);
  document.getElementById("mechanistic-score").textContent = best.mechanistic.toFixed(2);
  document.getElementById("intake-summary").textContent = intakeSummary(profile, tier, state);
  document.getElementById("review-queue").textContent = reviewQueue(best.action, tier);
  document.getElementById("audit-note").textContent = auditNote(best.action, best, state);

  renderBars("logit-bars", logits(after), { diverging: true });
  renderBars("attribution-bars", attribution(state), { diverging: false });
  renderActionTable(rows);
}

function intakeSummary(profile, tier, features) {
  const parts = [];
  parts.push(`${tier.toUpperCase()} tier`);
  if (features.sequenceRisk > 0.62) parts.push("elevated screening similarity");
  if (features.capabilityRisk > 0.62) parts.push("dual-use capability signal");
  if (features.synthesisScale > 0.62) parts.push("large synthesis scale");
  if (features.uncertainty > 0.62) parts.push("missing context");
  if (features.customerVerification < 0.45) parts.push("weak customer verification");
  if (features.documentationQuality < 0.45) parts.push("incomplete documentation");
  if (parts.length === 1) parts.push("low-risk educational or research request");
  return `${parts.join("; ")}.`;
}

function reviewQueue(action, tier) {
  if (action === "route_biosafety_review") return tier === "critical" ? "Priority human review" : "Standard biosafety review";
  if (action === "request_more_context") return "Awaiting customer metadata";
  if (action === "block_and_explain_safely") return "Blocked with safety explanation";
  if (action === "redact_and_offer_safe_alternative") return "Safe-completion response";
  return "No review required";
}

function auditNote(action, row, features) {
  const risk = riskIndex(features).toFixed(2);
  return `${ACTION_LABELS[action]} selected; risk index ${risk}; reward ${row.reward.toFixed(2)}; behavior ${row.behavior}.`;
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
      <span>${ACTION_LABELS[row.action]}</span>
      <span>${row.behavior}</span>
      <strong>${row.reward.toFixed(2)}</strong>
    `;
    root.appendChild(item);
  }
}

function mockFingerprintForProfile(profile) {
  return {
    regulated_sequence: "MOCK-FP-SCREEN-A17",
    scale_up_order: "MOCK-FP-SCALE-Q44",
    dual_use_protocol: "NO-SEQUENCE",
    ambiguous_order: "MOCK-FP-UNKNOWN-K03",
    benign_research: "MOCK-FP-BENIGN-R12",
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

function modelSnapshot(nextState = state) {
  const previous = state;
  state = { ...nextState };
  const profile = inferProfile(state);
  const rows = Object.keys(INTERVENTIONS)
    .map((action) => ({ action, ...score(profile, action) }))
    .sort((a, b) => b.reward - a.reward);
  const snapshot = {
    profile,
    tier: tierFor(state),
    recommendation: rows[0].action,
    reward: rows[0].reward,
    rows,
  };
  state = previous;
  return snapshot;
}

if (typeof globalThis !== "undefined") {
  globalThis.VerigradBioSafety = {
    PRESETS,
    INTERVENTIONS,
    actionSuitability,
    inferProfile,
    logits,
    modelSnapshot,
    riskIndex,
    tierFor,
  };
}

if (typeof document !== "undefined") {
  setupSliders();
  document.querySelectorAll(".preset").forEach((button) => {
    button.addEventListener("click", () => setPreset(button.dataset.preset));
  });
  render();
}
