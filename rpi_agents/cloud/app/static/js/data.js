// Data layer. Rendering code depends ONLY on this interface, never on where
// the data comes from — so replacing the demo fixtures with the real backend
// does not touch the UI.
//
// Two implementations behind one shape:
//   DemoSource  → reads contract fixtures from /dashboard/fixtures/{name}
//   LiveSource  → reads Wiktor's API under /v1/... with the session cookie
//
// Every method returns a plain contract object (or null when unavailable).
// A thrown DataError carries { code, message, status } for the UI to show.

export class DataError extends Error {
  constructor(code, message, status) {
    super(message || code);
    this.code = code;
    this.status = status || 0;
  }
}

async function getJSON(url, opts = {}) {
  let res;
  try {
    res = await fetch(url, { credentials: "same-origin", ...opts });
  } catch (e) {
    throw new DataError("NETWORK", "Could not reach the server", 0);
  }
  if (!res.ok) {
    let code = "ERROR";
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.error) {
        code = body.error.code || code;
        message = body.error.message || message;
      }
    } catch { /* non-JSON error body */ }
    throw new DataError(code, message, res.status);
  }
  return res.json();
}

//  demo source

class DemoSource {
  constructor() {
    this.isDemo = true;
  }

  async _fixture(name) {
    const body = await getJSON(`/dashboard/fixtures/${name}`);
    return body.data; // { schema_version, demo, fixture, data } → the raw contract object
  }

  async getDeviceStatus() {
    return this._fixture("device-status");
  }

  async getModelManifest() {
    return this._fixture("model-manifest");
  }

  async getLatestDecision() {
    return this._fixture("trigger"); // SNNDecision
  }

  async _demo(name) {
    const res = await fetch(`/static/demo/${name}.json`);
    return res.json();
  }

  async listEvents() {
    const doc = await this._demo("events");
    return doc.items || [];
  }

  async getEvent(id) {
    const doc = await this._demo("events");
    return (doc.items || []).find((e) => e.event_id === id) || null;
  }

  async getExperiments() {
    return this._demo("experiments"); // { runs: [...] }
  }

  async getEnergy() {
    return this._demo("energy"); // { sources: [...] }
  }
}

//  live source

// Which device the operator is looking at. In C1 there is no device picker yet,
// so we target the demo device id the backend seeds; C4/W wire real selection.
const LIVE_DEVICE_ID = "demo-pi";

class LiveSource {
  constructor() {
    this.isDemo = false;
  }

  async getDeviceStatus() {
    return getJSON(`/v1/devices/${LIVE_DEVICE_ID}/status`);
  }

  async getModelManifest() {
    // Exposed by the backend once a session/model is loaded; may be absent.
    return null;
  }

  async getLatestDecision() {
    return null;
  }

  async listEvents() {
    const body = await getJSON("/v1/events?limit=20");
    return body.items || [];
  }

  async getEvent(id) {
    return getJSON(`/v1/events/${id}`);
  }

  async getExperiments() {
    // No experiment-metrics endpoint in the v1 contract yet (W2 / Marcel).
    return null;
  }

  async getEnergy() {
    // No energy endpoint yet (A2 / Andrzej). Never fabricated as zeros.
    return null;
  }
}

export function createSource(mode) {
  return mode === "demo" ? new DemoSource() : new LiveSource();
}
