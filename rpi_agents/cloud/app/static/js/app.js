// Dashboard bootstrap (task C1): sign-in gate, explicit demo mode, tab shell.
// Data comes through the `data.js` source; this file only decides *what* to
// show and wires navigation. Per-tab rendering stays declarative below.

import { createSource } from "./data.js";
import * as auth from "./auth.js";
import { mountNetworkEditor } from "./network.js";
import { el, clear, kv, statusRow, stateLoading, stateEmpty, stateError } from "./ui.js";

const $ = (sel) => document.querySelector(sel);

const loginView = $("#login-view");
const appView = $("#app-view");
const demoBadge = $("#demo-badge");
const footerMode = $("#footer-mode");

let source = null; // the active DataSource (demo or live)
let mode = null;    // "demo" | "live"
const rendered = new Set(); // tabs already drawn (lazy, drawn once)

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

//  Network (board editor C2; neuron inspector / runtime LEDs are C3) 
async function renderNetwork(panel) {
  head(panel, "Network", "Lu.i board topology — draft editor. Neuron inspector and live signals arrive in C3.");
  const card = el("div", { class: "card" });
  panel.append(card);
  await mountNetworkEditor(card, { readonly: false });
}

//  Events (details & timeline are C4) 
async function renderEvents(panel) {
  head(panel, "Events", "Detection events with capture, vision and alarm timeline.");
  const card = el("div", { class: "card" }, [el("h3", { text: "Event timeline" })]);
  card.append(stateLoading());
  panel.append(card);
  try {
    const events = await source.listEvents();
    clear(card);
    card.append(el("h3", { text: "Event timeline" }));
    if (!events.length) {
      card.append(stateEmpty("No events", source.isDemo ? "No demo events available." : "No events recorded yet."));
      return;
    }
    const table = el("table", {}, [
      el("thead", {}, el("tr", {}, [
        el("th", { text: "Event" }), el("th", { text: "Stage" }), el("th", { text: "Status" }),
      ])),
    ]);
    const tbody = el("tbody");
    for (const ev of events) {
      const status = ev.status || (ev.trigger ? "trigger" : "");
      tbody.append(el("tr", {}, [
        el("td", { text: ev.event_id || "—" }),
        el("td", { text: ev.stage || "SNN trigger" }),
        el("td", {}, el("span", { class: "tag muted", text: status || "—" })),
      ]));
    }
    table.append(tbody);
    card.append(table);
    card.append(el("p", { class: "topbar-meta", text: "Full event details land in task C4." }));
  } catch (err) {
    clear(card);
    card.append(el("h3", { text: "Event timeline" }));
    card.append(stateError("Could not load events", err.message));
  }
}

// Experiments (metrics are C4) 
async function renderExperiments(panel) {
  head(panel, "Experiments", "Split, seed, model/encoder hashes, FA/h and recall.");
  const card = el("div", { class: "card" }, [el("h3", { text: "Experiment results" })]);
  card.append(comingSoon("C4", "Metrics with confidence intervals"));
  panel.append(card);
}

//  Energy (measurements are C4 / A2) 
async function renderEnergy(panel) {
  head(panel, "Energy", "Measured or estimated power and energy per session.");
  const card = el("div", { class: "card" }, [el("h3", { text: "Power & energy" })]);
  // Honest: no measurement yet — never shown as zero.
  card.append(stateEmpty("No measurement", "Energy figures arrive with tasks A2 / C4."));
  panel.append(card);
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
