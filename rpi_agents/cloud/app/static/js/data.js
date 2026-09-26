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

// --------------------------------------------------------------- demo source

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

  async listEvents() {
    // No standalone "events" fixture in the contracts; C4 builds the real
    // timeline. In demo we surface the one scripted decision as a single row
    // so the pipeline is visible without inventing data.
    const decision = await this.getLatestDecision().catch(() => null);
    if (!decision) return [];
    return [{
      event_id: decision.event_id,
      stage: "SNN trigger",
      trigger: decision.trigger,
      status: decision.status,
      source_time_us: decision.source_time_us,
    }];
  }
}

// --------------------------------------------------------------- live source

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
}

export function createSource(mode) {
  return mode === "demo" ? new DemoSource() : new LiveSource();
}
