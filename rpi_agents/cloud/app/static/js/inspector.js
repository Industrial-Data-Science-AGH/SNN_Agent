// Neuron inspector (task C3): Signals / Parameters / Connections / Notes.
// The Signals tab plots Vmem over time with the threshold, units and the
// calibration status — from the same runtime frames that drive the LEDs and the
// raster, so all three agree on neuron and time.

const SVGNS = "http://www.w3.org/2000/svg";

function e(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
function s(tag, attrs = {}) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  return n;
}

export function mountInspector(container, runtime, editor) {
  let neuronId = null;
  let tab = "signals";

  container.classList.add("inspector");
  const header = e("div", { class: "insp-head" });
  const tabsBar = e("div", { class: "insp-tabs" });
  const bodyEl = e("div", { class: "insp-body" });
  container.replaceChildren(header, tabsBar, bodyEl);

  const TABS = ["signals", "parameters", "connections", "notes"];
  const tabBtns = {};
  for (const t of TABS) {
    const b = e("button", { class: "insp-tab", type: "button", text: cap(t) });
    b.addEventListener("click", () => { tab = t; renderTabs(); renderBody(); });
    tabBtns[t] = b;
    tabsBar.append(b);
  }

  function cap(x) { return x[0].toUpperCase() + x.slice(1); }

  function renderTabs() {
    for (const t of TABS) tabBtns[t].classList.toggle("active", t === tab);
  }

  function renderHeader() {
    header.replaceChildren();
    if (!neuronId) { header.append(e("div", { class: "insp-empty", text: "Select a neuron on the canvas." })); return; }
    const frame = runtime.currentFrame;
    const nrn = frame?.neurons?.find((n) => n.neuron_id === neuronId);
    const active = nrn ? nrn.v_mem > (nrn.v_reset ?? 0) + 0.05 : false;
    header.append(
      e("div", { class: "insp-title" }, [
        e("strong", { text: `Neuron ${neuronId.toUpperCase()}` }),
        e("span", { class: `insp-dot ${active ? "ok" : "muted"}` }),
        e("span", { class: "insp-state", text: active ? "Active" : "Idle" }),
      ]),
    );
  }

  function renderBody() {
    bodyEl.replaceChildren();
    if (!neuronId) return;
    if (tab === "signals") renderSignals();
    else if (tab === "parameters") renderParameters();
    else if (tab === "connections") renderConnections();
    else renderNotes();
  }

  function renderSignals() {
    const meta = runtime.meta || {};
    const wrap = e("div");
    wrap.append(e("div", { class: "insp-label", text: `Vmem [${meta.potential_unit || "a.u."}]` }));
    const chart = s("svg", { class: "vmem-chart", viewBox: "0 0 320 180", preserveAspectRatio: "none" });
    wrap.append(chart);
    wrap.append(e("div", { class: "vmem-legend" }, [
      legend("vmem", "Vmem"), legend("thresh", "Threshold"), legend("spike", "Spikes"),
    ]));
    const cal = meta.calibration || {};
    wrap.append(e("div", { class: "insp-calib" }, [
      e("span", { text: "Calibration:" }),
      e("span", { class: `tag ${cal.status === "verified" ? "ok" : "warn"}`, text: cal.label || "Unverified" }),
    ]));
    bodyEl.append(wrap);
    drawVmem(chart);
  }

  function drawVmem(chart) {
    const frames = runtime.framesUpTo();
    const series = frames.map((f) => {
      const n = f.neurons.find((x) => x.neuron_id === neuronId);
      return { t: f.t ?? 0, v: n ? n.v_mem : 0, spiked: n?.spiked };
    });
    const meta = runtime.meta || {};
    const dur = meta.duration_s || (series.length ? series[series.length - 1].t : 1) || 1;
    const vth = meta.v_threshold ?? 1;
    let lo = Math.min(vth, ...series.map((p) => p.v));
    let hi = Math.max(vth, ...series.map((p) => p.v));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.15 || 0.2;
    lo -= pad; hi += pad;
    const PL = 34, PR = 6, PT = 8, PB = 22, W = 320, H = 180;
    const xw = W - PL - PR, yh = H - PT - PB;
    const X = (t) => PL + (t / dur) * xw;
    const Y = (v) => PT + (1 - (v - lo) / (hi - lo)) * yh;

    chart.replaceChildren();
    // axes
    chart.append(s("line", { x1: PL, y1: PT, x2: PL, y2: PT + yh, class: "axis" }));
    chart.append(s("line", { x1: PL, y1: PT + yh, x2: PL + xw, y2: PT + yh, class: "axis" }));
    for (const tv of [0, dur / 2, dur]) {
      const tx = X(tv);
      chart.append(s("line", { x1: tx, y1: PT + yh, x2: tx, y2: PT + yh + 3, class: "axis" }));
      const lab = s("text", { x: tx, y: H - 6, class: "axis-label", "text-anchor": "middle" });
      lab.textContent = Math.round(tv); chart.append(lab);
    }
    for (const vv of [lo + pad, vth, hi - pad]) {
      const yy = Y(vv);
      const lab = s("text", { x: PL - 4, y: yy + 3, class: "axis-label", "text-anchor": "end" });
      lab.textContent = vv.toFixed(1); chart.append(lab);
    }
    // threshold
    chart.append(s("line", { x1: PL, y1: Y(vth), x2: PL + xw, y2: Y(vth), class: "vmem-thresh" }));
    // vmem line
    if (series.length) {
      const d = series.map((p, i) => `${i ? "L" : "M"} ${X(p.t).toFixed(1)} ${Y(p.v).toFixed(1)}`).join(" ");
      chart.append(s("path", { d, class: "vmem-line" }));
    }
    // spike ticks
    for (const p of series) if (p.spiked) {
      chart.append(s("line", { x1: X(p.t), y1: PT + yh - 10, x2: X(p.t), y2: PT + yh, class: "vmem-spike" }));
    }
    // time cursor
    const tNow = runtime.currentTime;
    chart.append(s("line", { x1: X(tNow), y1: PT, x2: X(tNow), y2: PT + yh, class: "vmem-cursor" }));
  }

  function legend(cls, text) {
    return e("span", { class: "vmem-legend-item" }, [e("span", { class: `vmem-swatch ${cls}` }), text]);
  }

  function renderParameters() {
    const frame = runtime.currentFrame;
    const n = frame?.neurons?.find((x) => x.neuron_id === neuronId) || {};
    const meta = runtime.meta || {};
    bodyEl.append(row("Threshold", n.v_threshold ?? meta.v_threshold, meta.potential_unit));
    bodyEl.append(row("Reset", n.v_reset ?? meta.v_reset, meta.potential_unit));
    bodyEl.append(row("Vmem (now)", n.v_mem != null ? n.v_mem.toFixed(3) : null, meta.potential_unit));
    // tau_mem / tau_syn are board potentiometers, not in the frame stream:
    bodyEl.append(row("tau mem", null, "ms"));
    bodyEl.append(row("tau syn", null, "ms"));
    bodyEl.append(e("p", { class: "insp-note", text: "tau values come from the board/model, not the runtime stream (arrive with the model manifest)." }));
  }

  function row(label, value, unit) {
    const isNA = value == null || value === "";
    return e("div", { class: "insp-row" }, [
      e("span", { class: "insp-row-label", text: label }),
      isNA ? e("span", { class: "na", text: "Not available" })
           : e("span", { class: "insp-row-value", text: `${value}${unit ? " " + unit : ""}` }),
    ]);
  }

  function renderConnections() {
    const conns = editor.connectionsOf(neuronId);
    if (!conns.length) { bodyEl.append(e("div", { class: "insp-empty", text: "No connections." })); return; }
    for (const c of conns) {
      bodyEl.append(e("div", { class: "insp-row" }, [
        e("span", { class: `tag ${c.kind === "excitatory" ? "ok" : "alarm"}`, text: c.kind }),
        e("span", { class: "insp-row-value", text: `${c.direction === "out" ? "→" : "←"} ${c.other.toUpperCase()}` }),
      ]));
    }
  }

  function renderNotes() {
    const key = `snn-notes-${neuronId}`;
    let saved = "";
    try { saved = localStorage.getItem(key) || ""; } catch { /* private mode */ }
    const ta = e("textarea", { class: "insp-notes", rows: "6", placeholder: "Notes for this neuron (kept in this browser only)." });
    ta.value = saved;
    ta.addEventListener("input", () => { try { localStorage.setItem(key, ta.value); } catch { /* ignore */ } });
    bodyEl.append(ta);
  }

  renderTabs();
  renderHeader();
  renderBody();

  return {
    selectNeuron(id) { neuronId = id; renderHeader(); renderBody(); },
    update() { renderHeader(); if (tab === "signals") renderBody(); }, // live redraw of the chart
  };
}
