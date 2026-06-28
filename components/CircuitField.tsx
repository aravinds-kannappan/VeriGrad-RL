"use client";

import { useEffect, useRef } from "react";

/**
 * Animated neural-circuit background: drifting nodes, edges between near
 * neighbours, and signal pulses travelling along them. Thematic (circuits,
 * activations) and deliberately faint so it sits behind content. Honours
 * prefers-reduced-motion by rendering a single static frame.
 */
export default function CircuitField() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx0 = cv.getContext("2d");
    if (!ctx0) return;
    const el: HTMLCanvasElement = cv;
    const c: CanvasRenderingContext2D = ctx0;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const TEAL = "15, 118, 110";
    const INK = "11, 31, 51";

    let w = 0;
    let h = 0;
    let dpr = 1;
    type Node = { x: number; y: number; vx: number; vy: number };
    let nodes: Node[] = [];
    let edges: Array<[number, number]> = [];
    type Pulse = { e: number; t: number; speed: number };
    let pulses: Pulse[] = [];

    function build() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      el.width = w * dpr;
      el.height = h * dpr;
      el.style.width = w + "px";
      el.style.height = h + "px";
      c.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(64, Math.max(26, Math.round((w * h) / 28000)));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.12,
      }));
      computeEdges();
      pulses = edges
        .map((_, i) => i)
        .sort(() => Math.random() - 0.5)
        .slice(0, Math.round(edges.length * 0.18))
        .map((e) => ({ e, t: Math.random(), speed: 0.0016 + Math.random() * 0.0026 }));
    }

    const LINK = 168;
    function computeEdges() {
      edges = [];
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          if (dx * dx + dy * dy < LINK * LINK) edges.push([i, j]);
        }
      }
    }

    function frame() {
      c.clearRect(0, 0, w, h);

      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < -20) n.x = w + 20;
        if (n.x > w + 20) n.x = -20;
        if (n.y < -20) n.y = h + 20;
        if (n.y > h + 20) n.y = -20;
      }
      computeEdges();

      for (const [a, b] of edges) {
        const dx = nodes[a].x - nodes[b].x;
        const dy = nodes[a].y - nodes[b].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const alpha = (1 - dist / LINK) * 0.16;
        c.strokeStyle = `rgba(${INK}, ${alpha})`;
        c.lineWidth = 1;
        c.beginPath();
        c.moveTo(nodes[a].x, nodes[a].y);
        c.lineTo(nodes[b].x, nodes[b].y);
        c.stroke();
      }

      for (const n of nodes) {
        c.fillStyle = `rgba(${TEAL}, 0.5)`;
        c.beginPath();
        c.arc(n.x, n.y, 1.7, 0, Math.PI * 2);
        c.fill();
      }

      if (!reduce) {
        for (const p of pulses) {
          p.t += p.speed;
          if (p.t > 1) {
            p.t = 0;
            p.e = Math.floor(Math.random() * edges.length);
          }
          const edge = edges[p.e];
          if (!edge) continue;
          const [a, b] = edge;
          const x = nodes[a].x + (nodes[b].x - nodes[a].x) * p.t;
          const y = nodes[a].y + (nodes[b].y - nodes[a].y) * p.t;
          c.fillStyle = `rgba(${TEAL}, ${0.6 * (1 - Math.abs(0.5 - p.t) * 1.4)})`;
          c.beginPath();
          c.arc(x, y, 2.4, 0, Math.PI * 2);
          c.fill();
        }
      }

      raf = requestAnimationFrame(frame);
    }

    let raf = 0;
    build();
    if (reduce) {
      frame();
      cancelAnimationFrame(raf);
    } else {
      raf = requestAnimationFrame(frame);
    }

    let resizeTimer: ReturnType<typeof setTimeout>;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(build, 180);
    };
    const onVisibility = () => {
      if (document.hidden) cancelAnimationFrame(raf);
      else if (!reduce) raf = requestAnimationFrame(frame);
    };
    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return <canvas ref={ref} className="circuit-field" aria-hidden="true" />;
}
