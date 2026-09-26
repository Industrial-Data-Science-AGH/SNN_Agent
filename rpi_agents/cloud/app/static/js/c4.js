// Events, Experiments and Energy panels (task C4).
// All values come through the data source; missing values render as
// "Not available" and are NEVER shown as zero. Demo data is tagged. The
// experiment filter selects ONE run so metrics from different models / datasets
// / energy boundaries are never mixed.

import { el, clear, kv, stateLoading, stateEmpty, stateError } from "./ui.js";

const short = (h) => (typeof h === "string" && h.startsWith("sha256:") ? h.slice(7, 19) + "…" : h ?? "—");
const NA = () => el("span", { class: "na", text: "Not available" });

function demoTag(isDemo) {
  return isDemo ? el("span", { class: "tag warn", text: "Demo data" }) : null;
}

//  Events

const EVENT_STATUS = {
  alarm_confirmed: "alarm", review_required: "warn", no_alarm: "ok",
  analyzing: "muted", photo_requested: "muted", failed: "alarm",
};

export async function mountEvents(panel, source) {
  const layout = el("div", { class: "c4-master" });
  const listCol = el("div", { class: "c4-list card" }, [el("h3", { text: "Events" })]);
  const detailCol = el("div", { class: "c4-detail card" }, [stateEmpty("Select an event", "Pick an event to see its details.")]);
  layout.append(listCol, detailCol);
  panel.append(layout);

  const listHost = el("div");
  listCol.append(stateLoading());
  let events = [];
  try {
    events = await source.listEvents();
  } catch (err) {
    clear(listCol); listCol.append(el("h3", { text: "Events" }), stateError("Could not load events", err.message));
    return;
  }
  clear(listCol);
  listCol.append(el("div", { class: "c4-list-head" }, [el("h3", { text: "Events" }), demoTag(source.isDemo)]), listHost);
  if (!events.length) { listHost.append(stateEmpty("No events", "Nothing recorded yet.")); return; }

  const rows = [];
  events.forEach((ev) => {
    const row = el("button", { class: "c4-event-row", type: "button" }, [
      el("span", { class: "c4-event-id", text: ev.event_id }),
      el("span", { class: `tag ${EVENT_STATUS[ev.status] || "muted"}`, text: (ev.status || "").replace(/_/g, " ") || "—" }),
    ]);
    row.addEventListener("click", () => {
      rows.forEach((r) => r.classList.remove("active"));
      row.classList.add("active");
      renderEventDetail(detailCol, ev, source);
    });
    rows.push(row);
    listHost.append(row);
  });
  rows[0].click();
}

function stage(label, when, status, statusCls) {
  return el("div", { class: "c4-stage" }, [
    el("span", { class: `c4-stage-dot ${statusCls || "muted"}` }),
    el("span", { class: "c4-stage-label", text: label }),
    when ? el("span", { class: "c4-stage-time", text: when }) : el("span", { class: "na", text: "Not available" }),
    status ? el("span", { class: "c4-stage-status", text: status }) : null,
  ]);
}

function triState(v) {
  if (v === true) return { text: "Yes", cls: "ok" };
  if (v === false) return { text: "No", cls: "muted" };
  return { text: "Unknown", cls: "warn" };
}

function renderEventDetail(host, ev, source) {
  clear(host);
  const d = ev.decision || {};
  const v = ev.vision || null;
  const capture = (ev.commands || []).find((c) => c.type === "capture");
  const alarm = (ev.commands || []).find((c) => c.type === "alarm");

  host.append(el("div", { class: "c4-detail-head" }, [
    el("h3", { text: ev.event_id }),
    el("span", { class: `tag ${EVENT_STATUS[ev.status] || "muted"}`, text: (ev.status || "").replace(/_/g, " ") }),
  ]));

  // photo
  const photo = el("div", { class: "c4-photo" });
  if (!v || !v.image_id) {
    photo.append(el("div", { class: "c4-photo-empty" }, NA()));
  } else if (source.isDemo) {
    photo.append(el("div", { class: "c4-photo-empty", text: "Image available in live mode only (demo has no bytes)." }));
  } else {
    photo.append(el("img", { class: "c4-photo-img", src: `/v1/events/${ev.event_id}/images/0`, alt: `Event ${ev.event_id}` }));
  }
  host.append(photo);

  // error banner
  const errored = ev.status === "failed" || v?.status === "error" || (ev._acks || []).some((a) => a.status === "failed");
  if (errored) {
    const code = v?.error_code || (ev._acks || []).find((a) => a.status === "failed")?.error_code || "unknown";
    host.append(el("div", { class: "c4-error", text: `Error state: ${code}` }));
  }

  // timeline
  const tl = el("div", { class: "c4-section" }, [el("h4", { text: "Timeline" })]);
  tl.append(stage("SNN trigger", d.source_time_us != null ? `${(d.source_time_us / 1e6).toFixed(3)} s` : null,
    d.trigger ? "triggered" : "no trigger", d.trigger ? "alarm" : "muted"));
  tl.append(stage("Capture", capture?.issued_at || null, capture ? "requested" : "—", capture ? "warn" : "muted"));
  tl.append(stage("Vision", v?.status || null, v ? v.status : "—", v?.status === "ok" ? "ok" : v?.status === "error" ? "alarm" : "muted"));
  tl.append(stage("Alarm", alarm?.issued_at || null, alarm ? "raised" : "—", alarm ? "alarm" : "muted"));
  host.append(tl);

  // detection (glass / person / authorization kept separate)
  const det = el("div", { class: "c4-section" }, [el("h4", { text: "Detection" })]);
  if (!v) {
    det.append(stateEmpty("No vision result", "Vision did not run for this event."));
  } else {
    const g = triState(v.glass_visible), p = triState(v.person_visible);
    const grid = el("div", { class: "kv" });
    for (const [label, st] of [["Glass visible", g], ["Person visible", p]]) {
      grid.append(el("dt", { text: label }));
      grid.append(el("dd", {}, el("span", { class: `tag ${st.cls}`, text: st.text })));
    }
    grid.append(el("dt", { text: "Authorization" }));
    grid.append(el("dd", {}, el("span", { class: "tag warn", text: v.authorization || "unknown" })));
    grid.append(el("dt", { text: "Image quality" }));
    grid.append(el("dd", { text: v.image_quality || "unknown" }));
    grid.append(el("dt", { text: "Observation" }));
    grid.append(el("dd", { text: v.observation || "" }));
    det.append(grid);
  }
  host.append(det);

  // decision (SNN)
  const dec = el("div", { class: "c4-section" }, [el("h4", { text: "SNN decision" })]);
  dec.append(kv([
    ["Trigger", d.trigger != null ? (d.trigger ? "Yes" : "No") : null],
    ["Score", d.score_kind === "unavailable" || d.score == null ? null : `${d.score} (${d.score_kind})`],
    ["Model", short(d.model_hash)],
    ["Encoder", short(d.encoder_hash)],
  ]));
  host.append(dec);

  // ACK
  const ack = el("div", { class: "c4-section" }, [el("h4", { text: "Command ACK" })]);
  const acks = ev._acks;
  if (!source.isDemo) {
    ack.append(stateEmpty("Not available", "ACK status is wired with the backend (W2)."));
  } else if (!acks || !acks.length) {
    ack.append(el("div", { class: "na", text: "No commands" }));
  } else {
    for (const a of acks) {
      ack.append(el("div", { class: "insp-row" }, [
        el("span", { class: "insp-row-label", text: a.command_id }),
        el("span", {}, el("span", { class: `tag ${a.status === "acknowledged" ? "ok" : "alarm"}`, text: a.error_code ? `${a.status} · ${a.error_code}` : a.status })),
      ]));
    }
  }
  host.append(ack);
}

//  Experiments

function metricLine(m) {
  if (!m || m.value == null) return NA();
  const ci = (m.ci_low != null && m.ci_high != null) ? ` (95% CI ${m.ci_low}–${m.ci_high})` : "";
  return el("span", { text: `${m.value}${ci}` });
}

export async function mountExperiments(panel, source) {
  const card = el("div", { class: "card" }, [el("div", { class: "c4-list-head" }, [el("h3", { text: "Experiment results" }), demoTag(source.isDemo)])]);
  panel.append(card);
  card.append(stateLoading());

  let doc;
  try { doc = await source.getExperiments(); }
  catch (err) { clear(card); card.append(stateError("Could not load experiments", err.message)); return; }

  clear(card);
  card.append(el("div", { class: "c4-list-head" }, [el("h3", { text: "Experiment results" }), demoTag(source.isDemo)]));
  const runs = doc?.runs || [];
  if (!runs.length) {
    card.append(stateEmpty("Not available", "Experiment-metrics API is pending (W2 / Marcel)."));
    return;
  }

  // filter — one run at a time; results are never mixed across models/datasets
  const select = el("select", { class: "select", "aria-label": "Result filter" });
  runs.forEach((r) => select.append(new Option(`${r.dataset} · ${r.split} · model ${short(r.model_hash)}`, r.run_id)));
  const filterRow = el("div", { class: "c4-filter" }, [el("span", { class: "toolbar-label", text: "Result" }), select]);
  const body = el("div");
  card.append(filterRow, body);

  const renderRun = (run) => {
    clear(body);
    body.append(el("div", { class: "c4-run-meta" }, [
      metaItem("Dataset", run.dataset), metaItem("Split", run.split),
      metaItem("Background", run.background_hours != null ? `${run.background_hours} h` : null),
      metaItem("Seed", run.seed), metaItem("Model", short(run.model_hash)), metaItem("Encoder", short(run.encoder_hash)),
    ]));
    body.append(el("div", { class: "c4-metrics-grid" }, [
      metricsCard("SNN metrics", run.snn),
      metricsCard("Whole-system metrics", run.system),
    ]));
    body.append(el("p", { class: "insp-note", text: "SNN and whole-system metrics are shown separately for one model/dataset — never combined." }));
  };
  select.addEventListener("change", () => renderRun(runs.find((r) => r.run_id === select.value)));
  renderRun(runs[0]);
}

function metaItem(label, value) {
  return el("div", { class: "c4-meta-item" }, [
    el("span", { class: "c4-meta-label", text: label }),
    value == null || value === "" ? NA() : el("span", { class: "c4-meta-value", text: String(value) }),
  ]);
}

function metricsCard(title, m) {
  const c = el("div", { class: "card c4-metric-card" }, [el("h4", { text: title })]);
  const dl = el("dl", { class: "kv" });
  dl.append(el("dt", { text: "FA/h" }), el("dd", {}, metricLine(m?.fa_per_h)));
  dl.append(el("dt", { text: "Recall" }), el("dd", {}, metricLine(m?.recall)));
  c.append(dl);
  return c;
}

//  Energy

export async function mountEnergy(panel, source) {
  const card = el("div", { class: "card" }, [el("div", { class: "c4-list-head" }, [el("h3", { text: "Power & energy" }), demoTag(source.isDemo)])]);
  panel.append(card);
  card.append(stateLoading());

  let doc;
  try { doc = await source.getEnergy(); }
  catch (err) { clear(card); card.append(stateError("Could not load energy", err.message)); return; }

  clear(card);
  card.append(el("div", { class: "c4-list-head" }, [el("h3", { text: "Power & energy" }), demoTag(source.isDemo)]));
  const sources = doc?.sources || [];
  if (!sources.length) {
    // No measurement — explicitly, never a zero.
    card.append(stateEmpty("No measurement", "Energy figures are not available (A2 / Andrzej)."));
    return;
  }

  const grid = el("div", { class: "card-grid" });
  for (const s of sources) {
    const measured = s.source === "measured";
    const c = el("div", { class: "card c4-energy-card" }, [
      el("div", { class: "c4-list-head" }, [
        el("h4", { text: s.scope }),
        el("span", { class: `tag ${measured ? "ok" : "warn"}`, text: s.source }),
      ]),
      el("div", { class: "c4-energy-boundary", text: `Boundary: ${s.boundary}` }),
    ]);
    const dl = el("dl", { class: "kv" });
    dl.append(el("dt", { text: "Power" }), el("dd", {}, s.power_w == null ? NA() : el("span", { text: `${s.power_w} W` })));
    dl.append(el("dt", { text: "Energy" }), el("dd", {}, s.energy_j == null ? NA() : el("span", { text: `${s.energy_j} J · ${s.energy_wh} Wh` })));
    if (s.window_s) dl.append(el("dt", { text: "Window" }), el("dd", { text: `${s.window_s} s` }));
    c.append(dl);
    if (s.note) c.append(el("div", { class: "insp-note", text: s.note }));
    grid.append(c);
  }
  card.append(grid);
  card.append(el("p", { class: "insp-note", text: "Measured and estimated sources, and different measurement boundaries, are shown separately — never summed together." }));
}
