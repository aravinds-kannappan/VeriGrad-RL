// Copy-to-clipboard for terminal blocks + active-section highlight in the top nav.
(function () {
  "use strict";

  // Copy buttons.
  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy") || "";
      var done = function () {
        var prev = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = prev; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta); done();
      }
    });
  });

  // Scroll-spy: highlight the nav link for the section in view.
  var links = {};
  document.querySelectorAll(".bar-links a[href^='#']").forEach(function (a) {
    links[a.getAttribute("href").slice(1)] = a;
  });
  var sections = document.querySelectorAll("section[id]");
  if (!sections.length || !("IntersectionObserver" in window)) return;

  var visible = new Set();
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    });
    var best = null;
    sections.forEach(function (s) {
      if (visible.has(s.id)) {
        if (!best || s.getBoundingClientRect().top < best.getBoundingClientRect().top) best = s;
      }
    });
    Object.keys(links).forEach(function (id) { links[id].classList.remove("active"); });
    if (best && links[best.id]) links[best.id].classList.add("active");
  }, { rootMargin: "-60px 0px -70% 0px", threshold: 0 });

  sections.forEach(function (s) { obs.observe(s); });
})();
