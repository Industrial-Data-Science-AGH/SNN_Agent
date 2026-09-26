// Network editor (task C2): a layered SVG canvas of Lu.i boards with
// zoom / pan / Fit, dragging, neuron selection and excitatory/inhibitory
// connections. Boards are cloned from static/img/neuron.svg (its own layers:
// ports, potential LED, spike strip, selection halo) — never a plain circle.
//
// State is split into three isolated tiers so edits never touch a champion:
//   draft    editable topology (everything here mutates only this)
//   loaded   a loaded model — read-only reference (set by later tasks)
//   session  a running session — read-only (set in C3)
// Import/Export moves the DRAFT only, as data (JSON.parse — no code execution),
// and can never overwrite `loaded`/`session`.

const SVGNS = "http://www.w3.org/2000/svg";
const BW = 175;          // board width  (neuron.svg viewBox 210x120, same aspect)
const BH = 100;          // board height
const GAP_X = 55;        // grid gaps leave room for labels (no overlap)
const GAP_Y = 64;
const MAX_BOARDS = 50;

const KINDS = { excitatory: "excitatory", inhibitory: "inhibitory" };

//  pure model

function makeBoard(i) {
  return { id: `n${i + 1}`, label: `N${i + 1}`, x: 0, y: 0 };
}

/** Deterministic, layout-independent connection id (kept across relayout/drag). */
function connId(source, target) {
  return `${source}~${target}`;
}

/** A small default wiring so the excitatory/inhibitory view is populated. */
function defaultConnections(boards) {
  const conns = [];
  for (let i = 1; i < boards.length; i++) {
    const source = boards[i - 1].id;
    const target = boards[i].id;
    const kind = i % 3 === 0 ? KINDS.inhibitory : KINDS.excitatory;
    conns.push({ id: connId(source, target), source, target, kind });
  }
  return conns;
}

function gridLayout(boards) {
  const n = boards.length;
  if (!n) return;
  const cols = Math.ceil(Math.sqrt(n));
  boards.forEach((b, i) => {
    b.x = (i % cols) * (BW + GAP_X);
    b.y = Math.floor(i / cols) * (BH + GAP_Y);
  });
}

function validateTopology(obj) {
  if (!obj || typeof obj !== "object") throw new Error("Not a topology object");
  if (!Array.isArray(obj.boards)) throw new Error("Missing 'boards' array");
  if (obj.boards.length > MAX_BOARDS) throw new Error(`Too many boards (max ${MAX_BOARDS})`);
  const ids = new Set();
  for (const b of obj.boards) {
    if (typeof b?.id !== "string") throw new Error("Board without a string id");
    if (ids.has(b.id)) throw new Error(`Duplicate board id ${b.id}`);
    ids.add(b.id);
  }
  const conns = Array.isArray(obj.connections) ? obj.connections : [];
  for (const c of conns) {
    if (!ids.has(c?.source) || !ids.has(c?.target)) throw new Error("Connection references an unknown board");
    if (c.kind !== KINDS.excitatory && c.kind !== KINDS.inhibitory) throw new Error("Connection has an invalid kind");
  }
  // Return a clean copy — only known fields, nothing executable.
  return {
    boards: obj.boards.map((b) => ({
      id: String(b.id),
      label: String(b.label ?? b.id),
      x: Number(b.x) || 0,
      y: Number(b.y) || 0,
    })),
    connections: conns.map((c) => ({
      id: connId(c.source, c.target),
      source: String(c.source),
      target: String(c.target),
      kind: c.kind,
    })),
  };
}

// editor

export async function mountNetworkEditor(container, { readonly = false } = {}) {
  const state = {
    draft: { boards: [], connections: [] },
    loaded: null,   // read-only reference — never overwritten by import
    session: null,  // read-only — set in C3
    selectedId: null,
    view: { x: -40, y: -40, w: 900, h: 560 }, // viewBox
  };

  // DOM scaffold 
  container.innerHTML = "";
  const editor = elh("div", { class: "net-editor" });
  const toolbar = buildToolbar();
  const canvasWrap = elh("div", { class: "net-canvas-wrap" });
  const svg = document.createElementNS(SVGNS, "svg");
  svg.classList.add("net-canvas");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Lu.i board network");
  const defs = document.createElementNS(SVGNS, "defs");
  const gEdges = document.createElementNS(SVGNS, "g");
  const gBoards = document.createElementNS(SVGNS, "g");
  const emptyMsg = elh("div", { class: "net-empty" },
    "No boards. Set board count or import a topology.");
  svg.append(defs, gEdges, gBoards);
  canvasWrap.append(svg, emptyMsg);
  editor.append(toolbar.root, canvasWrap, buildLegend(), toolbar.readout);
  container.append(editor);

  //  neuron.svg symbol (injected once) 
  await injectSymbol(defs);

  //  state helpers 
  function applyView() {
    const { x, y, w, h } = state.view;
    svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  }

  function setBoardCount(n) {
    n = Math.max(0, Math.min(MAX_BOARDS, Math.floor(n) || 0));
    const boards = state.draft.boards;
    if (n < boards.length) {
      const kept = new Set(boards.slice(0, n).map((b) => b.id));
      boards.length = n;
      state.draft.connections = state.draft.connections.filter(
        (c) => kept.has(c.source) && kept.has(c.target));
      if (state.selectedId && !kept.has(state.selectedId)) select(null);
    } else {
      for (let i = boards.length; i < n; i++) boards.push(makeBoard(i));
      if (!state.draft.connections.length) state.draft.connections = defaultConnections(boards);
      else {
        // extend wiring for the newly added boards, keep existing ids
        const have = new Set(state.draft.connections.map((c) => c.id));
        for (const c of defaultConnections(boards)) if (!have.has(c.id)) state.draft.connections.push(c);
      }
    }
    gridLayout(boards);
    toolbar.countInput.value = String(boards.length);
    render();
    fit();
  }

  function select(id) {
    state.selectedId = id;
    gBoards.querySelectorAll(".board").forEach((g) =>
      g.classList.toggle("selected", g.dataset.id === id));
    const board = state.draft.boards.find((b) => b.id === id);
    toolbar.readout.textContent = board
      ? `Selected ${board.label} · ${state.draft.connections.filter((c) => c.source === id || c.target === id).length} connection(s)`
      : `${state.draft.boards.length} board(s) · draft topology`;
  }

  //  rendering 
  function boardCenter(b) { return { cx: b.x + BW / 2, cy: b.y + BH / 2 }; }

  function render() {
    const boards = state.draft.boards;
    emptyMsg.hidden = boards.length > 0;

    // edges
    gEdges.replaceChildren();
    const byId = Object.fromEntries(boards.map((b) => [b.id, b]));
    for (const c of state.draft.connections) {
      const s = byId[c.source], t = byId[c.target];
      if (!s || !t) continue;
      const { cx: sx, cy: sy } = boardCenter(s);
      const { cx: tx, cy: ty } = boardCenter(t);
      const path = document.createElementNS(SVGNS, "path");
      const mx = (sx + tx) / 2;
      path.setAttribute("d", `M ${sx} ${sy} C ${mx} ${sy}, ${mx} ${ty}, ${tx} ${ty}`);
      path.setAttribute("class", `edge edge-${c.kind}`);
      path.dataset.id = c.id;
      gEdges.append(path);
    }

    // boards
    gBoards.replaceChildren();
    for (const b of boards) {
      const g = document.createElementNS(SVGNS, "g");
      g.setAttribute("class", "board");
      g.dataset.id = b.id;
      g.setAttribute("transform", `translate(${b.x} ${b.y})`);
      const use = document.createElementNS(SVGNS, "use");
      use.setAttribute("href", "#lui-board");
      use.setAttribute("width", BW);
      use.setAttribute("height", BH);
      const label = document.createElementNS(SVGNS, "text");
      label.setAttribute("class", "board-label");
      label.setAttribute("x", BW / 2);
      label.setAttribute("y", BH + 18);
      label.setAttribute("text-anchor", "middle");
      label.textContent = b.label;
      g.append(use, label);
      if (b.id === state.selectedId) g.classList.add("selected");
      gBoards.append(g);
    }
  }

  //  zoom / pan / fit 
  function clientToCanvas(clientX, clientY) {
    const pt = svg.createSVGPoint();
    pt.x = clientX; pt.y = clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  function fit() {
    const boards = state.draft.boards;
    if (!boards.length) { state.view = { x: -40, y: -40, w: 900, h: 560 }; applyView(); return; }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const b of boards) {
      minX = Math.min(minX, b.x); minY = Math.min(minY, b.y);
      maxX = Math.max(maxX, b.x + BW); maxY = Math.max(maxY, b.y + BH + 24); // +label
    }
    const pad = 60;
    state.view = { x: minX - pad, y: minY - pad, w: (maxX - minX) + pad * 2, h: (maxY - minY) + pad * 2 };
    applyView();
  }

  function zoomBy(factor, centerClient) {
    const c = centerClient ? clientToCanvas(centerClient.x, centerClient.y)
      : { x: state.view.x + state.view.w / 2, y: state.view.y + state.view.h / 2 };
    const nw = clamp(state.view.w * factor, 200, 8000);
    const nh = clamp(state.view.h * factor, 120, 5000);
    state.view.x = c.x - (c.x - state.view.x) * (nw / state.view.w);
    state.view.y = c.y - (c.y - state.view.y) * (nh / state.view.h);
    state.view.w = nw; state.view.h = nh;
    applyView();
  }

  // interactions 
  let drag = null; // { id, board } dragging a board; or { pan:true } panning
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomBy(e.deltaY > 0 ? 1.1 : 0.9, { x: e.clientX, y: e.clientY });
  }, { passive: false });

  svg.addEventListener("pointerdown", (e) => {
    const boardEl = e.target.closest?.(".board");
    if (boardEl && !readonly) {
      const board = state.draft.boards.find((b) => b.id === boardEl.dataset.id);
      drag = { id: board.id, board, moved: false, start: clientToCanvas(e.clientX, e.clientY), ox: board.x, oy: board.y };
    } else {
      drag = { pan: true, start: { x: e.clientX, y: e.clientY }, vx: state.view.x, vy: state.view.y };
    }
    svg.setPointerCapture(e.pointerId);
  });

  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (drag.pan) {
      const scale = state.view.w / svg.clientWidth;
      state.view.x = drag.vx - (e.clientX - drag.start.x) * scale;
      state.view.y = drag.vy - (e.clientY - drag.start.y) * scale;
      applyView();
    } else {
      const p = clientToCanvas(e.clientX, e.clientY);
      drag.board.x = drag.ox + (p.x - drag.start.x);
      drag.board.y = drag.oy + (p.y - drag.start.y);
      drag.moved = Math.abs(p.x - drag.start.x) > 2 || Math.abs(p.y - drag.start.y) > 2;
      // move only this board + its edges; ids never change on layout
      const g = gBoards.querySelector(`.board[data-id="${drag.id}"]`);
      if (g) g.setAttribute("transform", `translate(${drag.board.x} ${drag.board.y})`);
      render();
    }
  });

  svg.addEventListener("pointerup", (e) => {
    if (drag && !drag.pan && !drag.moved) select(drag.id === state.selectedId ? null : drag.id);
    drag = null;
    try { svg.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  });

  // ---- toolbar wiring -----------------------------------------------------
  toolbar.countInput.addEventListener("change", () => setBoardCount(Number(toolbar.countInput.value)));
  toolbar.layoutBtn.addEventListener("click", () => { gridLayout(state.draft.boards); render(); fit(); });
  toolbar.fitBtn.addEventListener("click", fit);
  toolbar.zoomInBtn.addEventListener("click", () => zoomBy(0.83));
  toolbar.zoomOutBtn.addEventListener("click", () => zoomBy(1.2));

  toolbar.exportBtn.addEventListener("click", () => {
    const payload = { schema: "snn-topology-draft/1", ...state.draft };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "topology-draft.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  toolbar.importInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    toolbar.readout.classList.remove("is-error");
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);            // data only — never eval/Function
      const clean = validateTopology(parsed);      // draft only; loaded/session untouched
      state.draft = clean;
      select(null);
      toolbar.countInput.value = String(clean.boards.length);
      render();
      fit();
      toolbar.readout.textContent = `Imported ${clean.boards.length} board(s) into draft.`;
    } catch (err) {
      toolbar.readout.textContent = `Import failed: ${err.message}`;
      toolbar.readout.classList.add("is-error");
    } finally {
      e.target.value = ""; // allow re-importing the same file
    }
  });

  // ---- start --------------------------------------------------------------
  applyView();
  setBoardCount(8); // sensible default matching the reference
  return {
    setBoardCount,
    getDraft: () => structuredClone(state.draft), // payload for Patryk (copy, not live state)
  };
}

// ------------------------------------------------------------------ helpers

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function elh(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}

function buildToolbar() {
  const countInput = elh("input", { type: "number", min: "0", max: String(MAX_BOARDS), class: "net-count", "aria-label": "Board count" });
  countInput.value = "8";
  const layoutBtn = elh("button", { class: "btn btn-outline btn-sm", type: "button" }, "Auto-layout");
  const fitBtn = elh("button", { class: "btn btn-outline btn-sm", type: "button" }, "Fit");
  const zoomInBtn = elh("button", { class: "btn btn-outline btn-sm", type: "button", "aria-label": "Zoom in" }, "+");
  const zoomOutBtn = elh("button", { class: "btn btn-outline btn-sm", type: "button", "aria-label": "Zoom out" }, "−");
  const exportBtn = elh("button", { class: "btn btn-outline btn-sm", type: "button" }, "Export JSON");
  const importInput = elh("input", { type: "file", accept: "application/json,.json", id: "net-import", class: "net-file" });
  const importLabel = elh("label", { class: "btn btn-outline btn-sm", for: "net-import" }, "Import JSON");
  const readout = elh("div", { class: "net-readout" }, "");

  const root = elh("div", { class: "net-toolbar" }, [
    elh("span", { class: "toolbar-label" }, "Board count"),
    countInput,
    elh("span", { class: "toolbar-hint" }, "0 – 50"),
    layoutBtn, fitBtn,
    elh("span", { class: "net-zoom" }, [zoomOutBtn, zoomInBtn]),
    elh("span", { class: "net-spacer" }),
    exportBtn, importLabel, importInput,
  ]);
  return { root, countInput, layoutBtn, fitBtn, zoomInBtn, zoomOutBtn, exportBtn, importInput, readout };
}

function buildLegend() {
  return elh("div", { class: "net-legend" }, [
    legendItem("edge-excitatory", "Excitatory connection (excites)"),
    legendItem("edge-inhibitory", "Inhibitory connection (inhibits)"),
  ]);
}
function legendItem(cls, text) {
  return elh("span", { class: "net-legend-item" }, [elh("span", { class: `net-swatch ${cls}` }), text]);
}

async function injectSymbol(defs) {
  try {
    const res = await fetch("/static/img/neuron.svg");
    const text = await res.text();
    const doc = new DOMParser().parseFromString(text, "image/svg+xml");
    // Bring over the gradients (board-sheen, knob, …) and the board symbol.
    doc.querySelectorAll("defs > *").forEach((node) => defs.append(document.importNode(node, true)));
    const symbol = doc.querySelector("#lui-board");
    if (symbol) defs.append(document.importNode(symbol, true));
  } catch {
    // If the asset can't be loaded the canvas still works, just without art.
  }
}
