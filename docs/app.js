// Scroll-spy: highlight the table-of-contents entry for the section in view.
(function () {
  "use strict";
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (!links.length || !("IntersectionObserver" in window)) return;

  var byId = {};
  links.forEach(function (a) {
    var id = a.getAttribute("href").slice(1);
    byId[id] = a;
  });

  var visible = new Set();
  function update() {
    // Choose the topmost visible section.
    var best = null;
    document.querySelectorAll(".content > section").forEach(function (s) {
      if (visible.has(s.id)) {
        if (!best || s.getBoundingClientRect().top < best.getBoundingClientRect().top) best = s;
      }
    });
    links.forEach(function (a) { a.classList.remove("active"); });
    if (best && byId[best.id]) byId[best.id].classList.add("active");
  }

  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    });
    update();
  }, { rootMargin: "-86px 0px -65% 0px", threshold: 0 });

  document.querySelectorAll(".content > section").forEach(function (s) { obs.observe(s); });
})();
