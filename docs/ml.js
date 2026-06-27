// Real machine learning in the browser, on the real run data (window.AUP).
// (1) Logistic regression trained by gradient descent + an interactive predictor.
// (2) The kappa-paradox simulator. No libraries, no server.
(function () {
  "use strict";
  var D = window.AUP;
  var lossEl = document.getElementById("ml-loss");
  if (!D || !lossEl) return;

  var NF = D.features.length;
  var raw = D.rows;

  // Standardize features.
  var mean = [], std = [];
  for (var f = 0; f < NF; f++) {
    var m = 0; for (var i = 0; i < raw.length; i++) m += raw[i][f]; m /= raw.length;
    var v = 0; for (i = 0; i < raw.length; i++) { var d = raw[i][f] - m; v += d * d; }
    mean.push(m); std.push(Math.sqrt(v / raw.length) || 1);
  }
  function stdvec(vals) { return vals.map(function (x, f) { return (x - mean[f]) / std[f]; }); }
  var X = raw.map(function (r) { return stdvec(r.slice(0, NF)); });
  var Y = raw.map(function (r) { return r[NF]; });
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  var W = new Array(NF).fill(0), B = 0, losses = [];
  function reset() { W = new Array(NF).fill(0); B = 0; losses = []; }
  function epoch(lr) {
    var gw = new Array(NF).fill(0), gb = 0, loss = 0, n = X.length;
    for (var i = 0; i < n; i++) {
      var z = B; for (var f = 0; f < NF; f++) z += W[f] * X[i][f];
      var p = sigmoid(z), e = p - Y[i];
      for (f = 0; f < NF; f++) gw[f] += e * X[i][f];
      gb += e;
      loss += -(Y[i] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
    }
    for (f = 0; f < NF; f++) W[f] -= lr * gw[f] / n;
    B -= lr * gb / n;
    losses.push(loss / n);
  }
  function trainSync(lr, epochs) { reset(); for (var e = 0; e < epochs; e++) epoch(lr); }

  function renderLoss() {
    var w = 480, h = 150, pl = 46, pb = 26, pt = 12, pr = 12;
    var mn = Math.min.apply(null, losses), mx = Math.max.apply(null, losses);
    if (mx === mn) mx = mn + 1e-6;
    var pts = losses.map(function (l, i) {
      var x = pl + (i / (losses.length - 1 || 1)) * (w - pl - pr);
      var y = pt + (1 - (l - mn) / (mx - mn)) * (h - pt - pb);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    var svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%">';
    [0, 0.5, 1].forEach(function (t) {
      var y = pt + t * (h - pt - pb), val = mx - t * (mx - mn);
      svg += '<line x1="' + pl + '" y1="' + y + '" x2="' + (w - pr) + '" y2="' + y + '" stroke="#1b2940"/>' +
        '<text x="' + (pl - 6) + '" y="' + (y + 3) + '" fill="#6f8298" font-size="10" text-anchor="end">' + val.toFixed(2) + '</text>';
    });
    svg += '<polyline points="' + pts + '" fill="none" stroke="#45cdab" stroke-width="2"/>';
    svg += '<text x="' + ((pl + w - pr) / 2) + '" y="' + (h - 4) + '" fill="#6f8298" font-size="10" text-anchor="middle">epochs → (training log-loss)</text></svg>';
    lossEl.innerHTML = svg;
  }
  function renderCoef() {
    var el = document.getElementById("ml-coef"); if (!el) return;
    var mxw = Math.max.apply(null, W.map(Math.abs)) || 1, html = "";
    for (var f = 0; f < NF; f++) {
      var pos = W[f] >= 0, pct = Math.abs(W[f]) / mxw * 50;
      html += '<div class="coef-row"><span class="coef-lab">' + D.features[f] + '</span>' +
        '<div class="coef-track"><div class="coef-bar ' + (pos ? "pos" : "neg") + '" style="width:' +
        pct.toFixed(1) + '%;' + (pos ? "left:50%" : "right:50%") + '"></div></div>' +
        '<span class="coef-val">' + (pos ? "+" : "") + W[f].toFixed(2) + '</span></div>';
    }
    el.innerHTML = html;
  }
  function renderMetrics() {
    var el = document.getElementById("ml-metrics"); if (!el) return;
    el.innerHTML = 'converged log-loss <b>' + losses[losses.length - 1].toFixed(3) + '</b> · trained on <b>' +
      D.n + '</b> real samples (' + D.positives + ' deferred · ' + (100 * D.positives / D.n).toFixed(1) + '% base rate)';
  }
  function predictProb(model, csqa, intensity) {
    var xs = stdvec([intensity, model === "sonnet" ? 1 : 0, model === "haiku" ? 1 : 0, csqa ? 1 : 0]);
    var z = B; for (var f = 0; f < NF; f++) z += W[f] * xs[f];
    return sigmoid(z);
  }
  function renderPredict() {
    var m = document.getElementById("pred-model"); if (!m) return;
    var d = document.getElementById("pred-domain"), it = document.getElementById("pred-intensity");
    var iv = document.getElementById("pred-intensity-val");
    if (iv) iv.textContent = it.value;
    var p = predictProb(m.value, d.value === "csqa", parseFloat(it.value));
    document.getElementById("pred-out").textContent = (100 * p).toFixed(1) + "%";
    var g = document.getElementById("pred-gauge");
    if (g) g.style.width = Math.min(100, 100 * p / 0.5).toFixed(1) + "%";
  }

  var training = false;
  function trainAnimated() {
    if (training) return; training = true;
    var lrEl = document.getElementById("ml-lr"), epEl = document.getElementById("ml-epochs");
    var lr = parseFloat(lrEl ? lrEl.value : 0.5), epochs = parseInt(epEl ? epEl.value : 300, 10);
    reset();
    var done = 0, perFrame = Math.max(1, Math.ceil(epochs / 60));
    (function frame() {
      for (var k = 0; k < perFrame && done < epochs; k++) { epoch(lr); done++; }
      renderLoss();
      if (done < epochs) requestAnimationFrame(frame);
      else { renderCoef(); renderMetrics(); renderPredict(); training = false; }
    })();
  }

  var lrEl = document.getElementById("ml-lr"), epEl = document.getElementById("ml-epochs");
  if (lrEl) lrEl.addEventListener("input", function () { document.getElementById("ml-lr-val").textContent = lrEl.value; });
  if (epEl) epEl.addEventListener("input", function () { document.getElementById("ml-epochs-val").textContent = epEl.value; });
  var btn = document.getElementById("ml-train"); if (btn) btn.addEventListener("click", trainAnimated);
  ["pred-model", "pred-domain", "pred-intensity"].forEach(function (id) {
    var e = document.getElementById(id); if (e) e.addEventListener("input", renderPredict);
  });

  trainSync(lrEl ? parseFloat(lrEl.value) : 0.5, epEl ? parseInt(epEl.value, 10) : 300);
  renderLoss(); renderCoef(); renderMetrics(); renderPredict();

  // ---- kappa paradox ----
  function kappa(p, a) {
    var pA1 = p * a + (1 - p) * (1 - a);
    var Po = p * a * a + (1 - p) * (1 - a) * (1 - a) + p * (1 - a) * (1 - a) + (1 - p) * a * a;
    var Pe = pA1 * pA1 + (1 - pA1) * (1 - pA1);
    return { agree: Po, kappa: Pe >= 1 ? 0 : (Po - Pe) / (1 - Pe) };
  }
  function renderKappa() {
    var rate = document.getElementById("k-rate"); if (!rate) return;
    var acc = document.getElementById("k-acc");
    var p = parseFloat(rate.value) / 100, a = parseFloat(acc.value) / 100, r = kappa(p, a);
    document.getElementById("k-rate-val").textContent = parseFloat(rate.value).toFixed(1) + "%";
    document.getElementById("k-acc-val").textContent = parseFloat(acc.value).toFixed(1) + "%";
    var bars = document.getElementById("k-bars");
    if (bars) bars.innerHTML =
      '<div class="kb"><span>raw agreement</span><div class="kt"><div class="kf" style="width:' + (100 * r.agree).toFixed(1) + '%"></div></div><b>' + (100 * r.agree).toFixed(1) + '%</b></div>' +
      '<div class="kb"><span>Cohen’s κ</span><div class="kt"><div class="kf amber" style="width:' + (Math.max(0, r.kappa) * 100).toFixed(1) + '%"></div></div><b>' + r.kappa.toFixed(2) + '</b></div>';
    var verdict = document.getElementById("k-verdict");
    if (p <= 0.03 && r.agree > 0.9)
      verdict.innerHTML = "Raw agreement looks great, but κ has collapsed — the behavior is too rare to validate. <strong>This is exactly the trap that hid the spec-gaming detector bug.</strong>";
    else if (r.kappa > 0.8) verdict.innerHTML = "High κ — the graders agree well beyond chance. Trustworthy.";
    else verdict.innerHTML = "κ sits well below raw agreement — chance is doing much of the work.";
  }
  ["k-rate", "k-acc"].forEach(function (id) {
    var e = document.getElementById(id); if (e) e.addEventListener("input", renderKappa);
  });
  renderKappa();
})();
