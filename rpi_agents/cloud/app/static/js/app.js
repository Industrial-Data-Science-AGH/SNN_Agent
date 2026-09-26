// Dashboard bootstrap (task C1): sign-in gate, explicit demo mode, tab shell.
// Data comes through the `data.js` source; this file only decides *what* to
// show and wires navigation. Per-tab rendering stays declarative below.

import { createSource } from "./data.js";
import * as auth from "./auth.js";
import { mountNetworkEditor } from "./network.js";
import { createRuntime } from "./runtime.js";
import { mountInspector } from "./inspector.js";
import { mountRaster } from "./raster.js";
import { mountEvents, mountExperiments, mountEnergy } from "./c4.js";
import { el, clear, kv, statusRow, stateLoading, stateEmpty, stateError } from "./ui.js";

const $ = (sel) => document.querySelector(sel);

const loginView = $("#login-view");
const appView = $("#app-view");
const demoBadge = $("#demo-badge");
const footerMode = $("#footer-mode");

let source = null; // the active DataSource (demo or live)
let mode = null;    // "demo" | "live"
const rendered = new Set(); // tabs already drawn (lazy, drawn once)
let activeEditor = null; // the network editor, for the shared top toolbar

// The top chrome toolbar and the editor's own toolbar drive the SAME editor and
// stay in sync (board count, Fit). Top controls are live only on the Network tab.
function wireTopToolbar(editor) {
  const topCount = document.querySelector("#top-board-count");
  const topFit = document.querySelector("#top-fit");
  if (topCount && !topCount.dataset.wired) {
    topCount.dataset.wired = "1";
    topCount.replaceChildren(...Array.from({ length: 51 }, (_, i) => new Option(String(i), String(i))));
    topCount.addEventListener("change", () => activeEditor?.setBoardCount(Number(topCount.value)));
  }
  if (topFit && !topFit.dataset.wired) {
    topFit.dataset.wired = "1";
    topFit.addEventListener("click", () => activeEditor?.fit());
  }
  // editor → top: any count change (from either toolbar) reflects here
  editor.onCount((n) => { if (topCount) topCount.value = String(n); });
  if (topCount) topCount.value = String(editor.getCount());
}

function updateTopToolbar(tab) {
  const on = tab === "network" && !!activeEditor;
  const ids = ["#top-board-count", "#top-fit", "#top-live", "#top-replay", "#top-edit", "#top-start", "#top-pause"];
  for (const sel of ids) {
    const elm = document.querySelector(sel);
    if (elm) elm.disabled = !on;
  }
  const topCount = document.querySelector("#top-board-count");
  if (on && topCount) topCount.value = String(activeEditor.getCount());
}

//  entry points

async function start() {
  const session = await auth.currentSession();
  if (session) {
    enterApp("live");
  } else {
    showLogin();
  }
}

function showLogin() {
  appView.hidden = true;
  loginView.hidden = false;
  const username = $("#username");
  if (username) username.focus();
}

function enterApp(nextMode) {
  mode = nextMode;
  source = createSource(mode);
  loginView.hidden = true;
  appView.hidden = false;

  const isDemo = mode === "demo";
  demoBadge.hidden = !isDemo;
  footerMode.textContent = isDemo ? "Demo mode · sample data" : "Live · connected";

  rendered.clear();
  selectTab("network");
}

// --------------------------------------------------------------- sign-in form

function wireLogin() {
  const form = $("#login-form");
  const submit = $("#login-submit");
  const errorBox = $("#login-error");

  const showError = (msg) => {
    errorBox.textContent = msg;
    errorBox.hidden = false;
  };
  const clearError = () => { errorBox.hidden = true; errorBox.textContent = ""; };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    const username = $("#username").value.trim();
    const password = $("#password").value; // used once, never stored
    if (!username || !password) {
      showError("Enter username and password.");
      return;
    }
    submit.classList.add("is-loading");
    submit.disabled = true;
    try {
      await auth.login(username, password);
      $("#password").value = ""; // drop the secret from the DOM immediately
      enterApp("live");
    } catch (err) {
      showError(err.message || "Sign-in failed.");
      $("#password").select();
    } finally {
      submit.classList.remove("is-loading");
      submit.disabled = false;
    }
  });

  $("#demo-enter").addEventListener("click", () => enterApp("demo"));
}

// --------------------------------------------------------------- navigation

function wireShell() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => selectTab(btn.dataset.tab));
  });

  $("#logout-btn").addEventListener("click", async () => {
    if (mode === "live") await auth.logout();
    // Demo mode has no server session; both paths return to the sign-in gate.
    showLogin();
  });
}

function selectTab(tab) {
  document.querySelectorAll(".nav-item").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".panel").forEach((p) =>
    (p.hidden = p.dataset.panel !== tab));

  if (!rendered.has(tab)) {
    rendered.add(tab);
    renderTab(tab);
  }
  updateTopToolbar(tab);
}

//  tab rendering

const TABS = {
  network: renderNetwork,
  events: renderEvents,
  experiments: renderExperiments,
  energy: renderEnergy,
  device: renderDevice,
};

async function renderTab(tab) {
  const panel = document.querySelector(`.panel[data-panel="${tab}"]`);
  clear(panel);
  const fn = TABS[tab];
  if (fn) await fn(panel);
}

function head(panel, title, subtitle) {
  panel.append(el("div", { class: "panel-head" }, [
    el("h2", { text: title }),
    subtitle ? el("p", { text: subtitle }) : null,
  ]));
}

function comingSoon(task, what) {
  return stateEmpty(`${what}`, `Built in task ${task}.`);
}

//  Network (board editor C2 + runtime LEDs, inspector, raster, replay — C3)
async function renderNetwork(panel) {
  head(panel, "Network", "Lu.i board topology, live signals and neuron inspector.");

  // runtime controls: Live / Replay / Pause view, status, replay scrubber
  const controls = buildRuntimeControls(source.isDemo);
  panel.append(controls.root);

  // canvas + inspector side by side; raster below
  const canvasCard = el("div", { class: "card net-canvas-card" });
  const inspectorCard = el("div", { class: "card insp-card" });
  panel.append(el("div", { class: "net-layout" }, [canvasCard, inspectorCard]));
  const rasterCard = el("div", { class: "card" }, [el("h3", { text: "Spike raster" })]);
  const rasterHost = el("div");
  rasterCard.append(rasterHost);
  panel.append(rasterCard);

  const editor = await mountNetworkEditor(canvasCard, { readonly: false });
  activeEditor = editor;
  wireTopToolbar(editor);
  updateTopToolbar("network");
  const runtime = createRuntime(source.isDemo ? "demo" : "live");
  await runtime.load();
  const inspector = mountInspector(inspectorCard, runtime, editor);
  const raster = mountRaster(rasterHost, runtime, (t) => { switchMode("replay"); runtime.seek(t); });

  editor.onSelect((id) => inspector.selectNeuron(id));

  runtime.on("frame", (frame, idx) => {
    editor.applyFrame(frame);
    inspector.update();
    raster.update();
    controls.setTime(runtime.currentTime, runtime.meta?.duration_s);
    if (!controls.scrubbing) controls.scrubber.value = String(idx);
  });
  runtime.on("status", (st) => controls.setStatus(st));

  //  control wiring — panel runbar AND the top toolbar drive the same runtime
  const top = {
    live: document.querySelector("#top-live"),
    replay: document.querySelector("#top-replay"),
    edit: document.querySelector("#top-edit"),
    start: document.querySelector("#top-start"),
    pause: document.querySelector("#top-pause"),
  };
  let currentMode = "live"; // live | replay | edit
  let viewPaused = false;

  function paintMode() {
    const map = [
      ["live", top.live, controls.liveBtn],
      ["replay", top.replay, controls.replayBtn],
      ["edit", top.edit, null],
    ];
    for (const [m, tb, pb] of map) {
      const active = m === currentMode;
      for (const btn of [tb, pb]) {
        if (!btn) continue;
        btn.classList.toggle("btn-primary", active);
        btn.classList.toggle("btn-outline", !active);
      }
    }
  }
  function paintPaused() {
    controls.setPaused(viewPaused);
    if (top.pause) top.pause.textContent = viewPaused ? "Resume" : "Pause view";
  }

  function switchMode(mode) {
    currentMode = mode;
    viewPaused = mode === "edit";               // Edit freezes the view for editing
    controls.scrubber.disabled = mode !== "replay";
    paintMode();
    paintPaused();
    if (mode === "edit") runtime.pause();        // pauses the view only — never Stop
    else runtime.play({ mode });
  }
  function togglePause() {
    viewPaused = !viewPaused;
    paintPaused();
    // Pause view stops advancing the view only — it never stops the session.
    if (viewPaused) runtime.pause(); else runtime.resume();
  }

  top.live?.addEventListener("click", () => switchMode("live"));
  top.replay?.addEventListener("click", () => switchMode("replay"));
  top.edit?.addEventListener("click", () => switchMode("edit"));
  top.pause?.addEventListener("click", togglePause);
  top.start?.addEventListener("click", () => {          // Start: run from the beginning, live
    runtime.seek(runtime.frames?.[0]?.t ?? 0);
    switchMode("live");
  });

  controls.scrubber.max = String((runtime.meta?.frameCount || 1) - 1);
  controls.scrubber.addEventListener("input", () => {
    controls.scrubbing = true;
    const frames = runtime.frames || [];
    const idx = Math.min(frames.length - 1, Number(controls.scrubber.value));
    if (frames[idx]) runtime.seek(frames[idx].t);
  });
  controls.scrubber.addEventListener("change", () => { controls.scrubbing = false; });
  if (controls.lossBtn) {
    controls.lossBtn.addEventListener("click", () => {
      const on = controls.lossBtn.dataset.on !== "true";
      controls.lossBtn.dataset.on = String(on);
      controls.lossBtn.textContent = on ? "Restore connection" : "Simulate connection loss";
      runtime.simulateLoss(on);
    });
  }

  // default: live view, inspect a neuron so the chart is populated
  editor.select("n5");
  switchMode("live");
}

// The panel run-bar holds only what the fixed top chrome can't: connection
// status, the time readout, the replay scrubber and the demo loss toggle. The
// Live/Replay/Edit/Pause/Start buttons live in the top toolbar.
function buildRuntimeControls(isDemo) {
  const statusBadge = el("span", { class: "conn-badge", text: "Connecting…" });
  const timeLabel = el("span", { class: "conn-time", text: "0 s" });
  const scrubber = el("input", { type: "range", min: "0", max: "100", value: "0", class: "net-scrub", "aria-label": "Replay position" });
  scrubber.disabled = true;
  const lossBtn = isDemo
    ? el("button", { class: "btn btn-ghost btn-sm", type: "button", "data-on": "false", text: "Simulate connection loss" })
    : null;

  const root = el("div", { class: "runbar" }, [
    el("span", { class: "toolbar-label", text: "Signal" }),
    statusBadge, timeLabel,
    el("span", { class: "net-spacer" }),
    scrubber,
    lossBtn,
  ]);

  return {
    root, scrubber, lossBtn, scrubbing: false,
    setPaused() { /* Pause label lives on the top toolbar */ },
    setTime(t, dur) { timeLabel.textContent = dur ? `${t.toFixed(1)} / ${dur} s` : `${t.toFixed(1)} s`; },
    setStatus(st) {
      const map = { connected: ["ok", "Live"], stale: ["alarm", "Stale data"], reconnecting: ["warn", "Reconnecting…"] };
      const [cls, text] = map[st] || ["muted", st];
      statusBadge.className = `conn-badge ${cls}`;
      statusBadge.textContent = text;
    },
  };
}

//  Events (details: photo, timeline, glass/person/authorization, ACK, error)
async function renderEvents(panel) {
  head(panel, "Events", "Detection events with capture, vision and alarm timeline.");
  await mountEvents(panel, source);
}

//  Experiments (SNN vs whole-system metrics; filter never mixes runs)
async function renderExperiments(panel) {
  head(panel, "Experiments", "Split, seed, model/encoder hashes, FA/h with confidence interval and recall.");
  await mountExperiments(panel, source);
}

//  Energy (measured/estimated kept separate; missing = Not available, never zero)
async function renderEnergy(panel) {
  head(panel, "Energy", "Measured or estimated power and energy, with the measurement boundary.");
  await mountEnergy(panel, source);
}

//  Device (fully rendered in C1 from the DeviceStatus contract) 
async function renderDevice(panel) {
  head(panel, "Device", "Edge agent heartbeat, serial link, camera and outbox.");
  const card = el("div", { class: "card" }, [el("h3", { text: "Device status" })]);
  card.append(stateLoading());
  panel.append(card);
  try {
    const s = await source.getDeviceStatus();
    clear(card);
    card.append(el("h3", { text: "Device status" }));

    card.append(kv([
      ["Device", s.device_id],
      ["State", s.state],
      ["Agent version", s.agent_version],
      ["Session", s.session_id],
      ["Uptime (s)", s.uptime_s],
      ["Reported at", s.reported_at],
    ]));

    const serial = s.serial || {};
    const camera = s.camera || {};
    const outbox = s.outbox || {};

    const grid = el("div", { class: "card-grid" }, [
      el("div", { class: "card" }, [
        el("h3", { text: "Serial" }),
        statusRow(serial.connected ? "Connected" : "Disconnected", !!serial.connected,
          serial.last_frame_age_ms != null ? `last frame ${serial.last_frame_age_ms} ms ago` : null),
        kv([
          ["Frames", serial.frames], ["Gaps", serial.gaps], ["Rejected", serial.rejected],
          ["Stalls", serial.stalls], ["Reconnects", serial.reconnects],
        ]),
      ]),
      el("div", { class: "card" }, [
        el("h3", { text: "Camera" }),
        statusRow(camera.configured ? "Configured" : "Not configured", !!camera.configured),
      ]),
      el("div", { class: "card" }, [
        el("h3", { text: "Outbox" }),
        kv([["Pending", outbox.pending], ["Dead", outbox.dead], ["Dropped total", outbox.dropped_total]]),
      ]),
    ]);
    card.append(grid);
  } catch (err) {
    clear(card);
    card.append(el("h3", { text: "Device status" }));
    card.append(stateError("Could not load device status", err.message));
  }
}

// go

wireLogin();
wireShell();
start();
