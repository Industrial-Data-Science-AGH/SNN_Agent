// Spike raster (task C3): one row per neuron, a tick at every spike, drawn from
// the SAME runtime frames as the LEDs and the Vmem chart — so a tick and an LED
// flash always mark the same neuron at the same time. A cursor marks "now".

const SVGNS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  return n;
};

export function mountRaster(container, runtime, onSeek) {
  const svg = s("svg", { class: "raster", viewBox: "0 0 640 240", preserveAspectRatio: "none" });
  container.replaceChildren(svg);

  // clicking/scrubbing the raster seeks replay to that time
  if (onSeek) {
    svg.addEventListener("click", (e) => {
      const meta = runtime.meta || {};
      const dur = meta.duration_s || 1;
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      const PL = 34, PR = 8, W = 640, xw = W - PL - PR;
      const t = ((loc.x - PL) / xw) * dur;
      if (t >= 0 && t <= dur) onSeek(t);
    });
  }

  function update() {
    const meta = runtime.meta || {};
    const ids = meta.neuron_ids && meta.neuron_ids.length
      ? meta.neuron_ids
      : (runtime.currentFrame?.neurons || []).map((n) => n.neuron_id);
    const dur = meta.duration_s || (runtime.currentTime || 1) || 1;
    const frames = runtime.framesUpTo();

    const PL = 34, PR = 8, PT = 10, PB = 22, W = 640, H = 240;
    const xw = W - PL - PR, yh = H - PT - PB;
    const rows = ids.length || 1;
    const rowH = yh / rows;
    const X = (t) => PL + (t / dur) * xw;
    const rowY = (i) => PT + i * rowH + rowH / 2;

    svg.replaceChildren();
    // axes + neuron labels
    svg.append(s("line", { x1: PL, y1: PT, x2: PL, y2: PT + yh, class: "axis" }));
    svg.append(s("line", { x1: PL, y1: PT + yh, x2: PL + xw, y2: PT + yh, class: "axis" }));
    ids.forEach((id, i) => {
      const lab = s("text", { x: PL - 5, y: rowY(i) + 3, class: "axis-label", "text-anchor": "end" });
      lab.textContent = id.toUpperCase(); svg.append(lab);
    });
    for (const tv of [0, dur / 4, dur / 2, (dur * 3) / 4, dur]) {
      const lab = s("text", { x: X(tv), y: H - 6, class: "axis-label", "text-anchor": "middle" });
      lab.textContent = Math.round(tv); svg.append(lab);
    }
    // spikes
    const idIndex = Object.fromEntries(ids.map((id, i) => [id, i]));
    // bound the number of frames scanned so a long session stays cheap to draw
    const MAX_FRAMES = 1500;
    const step = frames.length > MAX_FRAMES ? Math.ceil(frames.length / MAX_FRAMES) : 1;
    for (let fi = 0; fi < frames.length; fi += step) {
      const f = frames[fi];
      for (const n of f.neurons || []) {
        if (!n.spiked) continue;
        const i = idIndex[n.neuron_id];
        if (i == null) continue;
        const x = X(f.t ?? 0);
        svg.append(s("line", { x1: x, y1: rowY(i) - rowH * 0.32, x2: x, y2: rowY(i) + rowH * 0.32, class: "raster-tick" }));
      }
    }
    // time cursor
    const cx = X(runtime.currentTime);
    svg.append(s("line", { x1: cx, y1: PT, x2: cx, y2: PT + yh, class: "raster-cursor" }));
  }

  return { update };
}
