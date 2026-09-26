// Runtime signal layer (task C3): supplies neuron frames to the LEDs, the
// inspector chart and the spike raster from ONE source, so they always agree
// on the same neuron/time. Nothing here is random — LEDs and spikes come from
// the runtime fields (v_mem, spiked), never a decorative animation or tau alone.
//
// Two implementations behind one interface:
//   DemoRuntime  → a deterministic golden replay (static/demo/neuron-frames.json)
//   LiveRuntime  → Server-Sent Events from Patryk's stream (P3), with reconnect
//
// Playback: play({mode})/pause()/resume()/seek(t)/stop(). "Pause view" calls
// pause() — it stops advancing the view and NEVER stops the session (no Stop is
// sent). Connection health is reported as connected | stale | reconnecting.

const STALE_MS = 3000;

class Emitter {
  constructor() { this._h = {}; }
  on(evt, cb) { (this._h[evt] ||= []).push(cb); return this; }
  emit(evt, ...a) { (this._h[evt] || []).forEach((cb) => cb(...a)); }
}

//  demo (golden)

class DemoRuntime extends Emitter {
  constructor() {
    super();
    this.isDemo = true;
    this.frames = [];
    this.meta = null;
    this.index = 0;
    this._timer = null;
    this._lost = false;       // simulated connection loss
    this._staleTimer = null;
  }

  async load() {
    const res = await fetch("/static/demo/neuron-frames.json");
    const doc = await res.json();
    this.frames = doc.frames || [];
    this.meta = {
      duration_s: doc.duration_s,
      dt_s: doc.dt_s,
      potential_unit: doc.potential_unit || "a.u.",
      v_threshold: doc.v_threshold ?? 1,
      v_reset: doc.v_reset ?? 0,
      neuron_ids: doc.neuron_ids || [],
      calibration: doc.calibration || { status: "unverified", label: "Unverified" },
      frameCount: this.frames.length,
      demo: true,
    };
    return this.meta;
  }

  get currentFrame() { return this.frames[this.index] || null; }
  get currentTime() { return this.currentFrame ? this.currentFrame.t : 0; }
  framesUpTo(index = this.index) { return this.frames.slice(0, index + 1); }

  _tick() {
    if (this._lost) return;                 // no frames while "disconnected"
    this.index = (this.index + 1) % this.frames.length; // live loops for the demo
    this._deliver();
  }

  _deliver() {
    this.emit("frame", this.currentFrame, this.index);
    this._markFresh();
  }

  _markFresh() {
    this.emit("status", "connected");
    clearTimeout(this._staleTimer);
    this._staleTimer = setTimeout(() => this.emit("status", "stale"), STALE_MS);
  }

  play({ mode = "live" } = {}) {
    this.mode = mode;
    this.stop();
    this._deliver();
    if (mode === "live") {
      // accelerated wall clock: one dt frame every 250 ms so the view is lively
      this._timer = setInterval(() => this._tick(), 250);
    }
    // replay is scrubber-driven: no auto-advance
  }

  pause() { clearInterval(this._timer); this._timer = null; }        // does NOT stop the session
  resume() { if (this.mode === "live" && !this._timer) this._timer = setInterval(() => this._tick(), 250); }
  stop() { clearInterval(this._timer); this._timer = null; }         // teardown of the view only

  seek(t) {
    // jump to the nearest frame and rebuild a consistent state (not just scroll)
    let best = 0, bestD = Infinity;
    this.frames.forEach((f, i) => { const d = Math.abs(f.t - t); if (d < bestD) { bestD = d; best = i; } });
    this.index = best;
    this._deliver();
  }

  // demo-only: exercise the Stale data + reconnect path without a backend
  simulateLoss(on) {
    this._lost = on;
    if (on) {
      clearTimeout(this._staleTimer);
      this.emit("status", "reconnecting");
      this._staleTimer = setTimeout(() => this.emit("status", "stale"), STALE_MS);
    } else {
      this._deliver(); // reconnected: fresh frame, consistent state
    }
  }
}

// ------------------------------------------------------------- live (SSE)

class LiveRuntime extends Emitter {
  constructor(sessionId) {
    super();
    this.isDemo = false;
    this.sessionId = sessionId;
    this.frames = [];
    this.meta = null;
    this.index = 0;
    this._es = null;
    this._staleTimer = null;
    this._retry = null;
  }

  async load() {
    // Meta is learned from the first frame; sensible defaults until then.
    this.meta = { potential_unit: "a.u.", v_threshold: 1, v_reset: 0, neuron_ids: [], calibration: { status: "unverified", label: "Unverified" }, frameCount: 0, demo: false };
    return this.meta;
  }

  get currentFrame() { return this.frames[this.index] || null; }
  get currentTime() { return this.currentFrame ? (this.currentFrame.t ?? 0) : 0; }
  framesUpTo(index = this.index) { return this.frames.slice(0, index + 1); }

  _connect() {
    this.emit("status", "reconnecting");
    try {
      this._es = new EventSource(`/v1/sessions/${this.sessionId}/telemetry`);
    } catch {
      this._scheduleReconnect();
      return;
    }
    this._es.addEventListener("snapshot", (e) => this._onFrame(e));
    this._es.onmessage = (e) => this._onFrame(e);
    this._es.onerror = () => { this._es?.close(); this._scheduleReconnect(); };
  }

  _onFrame(e) {
    let frame;
    try { frame = JSON.parse(e.data); } catch { return; } // data only, no eval
    this.frames.push(frame);
    this.index = this.frames.length - 1;
    if (this.meta && frame.neurons) this.meta.neuron_ids = frame.neurons.map((n) => n.neuron_id);
    this.emit("frame", frame, this.index);
    this.emit("status", "connected");
    clearTimeout(this._staleTimer);
    this._staleTimer = setTimeout(() => this.emit("status", "stale"), STALE_MS);
  }

  _scheduleReconnect() {
    this.emit("status", "stale");
    clearTimeout(this._retry);
    this._retry = setTimeout(() => this._connect(), 1000);
  }

  play() { this._connect(); }
  pause() { this._es?.close(); this._es = null; }   // stop consuming the stream; DO NOT stop the session
  resume() { if (!this._es) this._connect(); }
  stop() { clearTimeout(this._retry); this._es?.close(); this._es = null; }
  seek() { /* live has no seek; replay uses the demo/stored buffer */ }
}

export function createRuntime(mode, sessionId = "demo-session") {
  return mode === "demo" ? new DemoRuntime() : new LiveRuntime(sessionId);
}
