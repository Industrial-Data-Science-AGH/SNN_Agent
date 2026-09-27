// Small DOM helpers and shared UI states.
// Rendering-only: this module never fetches or knows where data comes from.

/** Create an element. `attrs.class`, `attrs.text`, `attrs.html`, dataset via data-*. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("data-")) node.setAttribute(key, value);
    else node[key] = value;
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** A spinner block for "loading" content areas. */
export function stateLoading(message = "Loading…") {
  return el("div", { class: "state" }, [
    el("div", { class: "spinner-lg" }),
    el("div", { class: "state-title", text: message }),
  ]);
}

/** An explicit empty state (never rendered as zeros or fake values). */
export function stateEmpty(title, detail) {
  return el("div", { class: "state" }, [
    el("div", { class: "state-title", text: title }),
    detail ? el("div", { text: detail }) : null,
  ]);
}

export function stateError(title, detail) {
  return el("div", { class: "state error" }, [
    el("div", { class: "state-title", text: title }),
    detail ? el("div", { text: detail }) : null,
  ]);
}

/** A key/value definition list from an array of [label, value] pairs. */
export function kv(pairs) {
  const dl = el("dl", { class: "kv" });
  for (const [label, value] of pairs) {
    dl.append(el("dt", { text: label }));
    const isNA = value == null || value === "";
    dl.append(el("dd", isNA ? { class: "na", text: "Not available" } : { text: String(value) }));
  }
  return dl;
}

export function statusRow(label, ok, detail) {
  const state = ok === true ? "ok" : ok === false ? "alarm" : "warn";
  return el("div", { class: "status-row" }, [
    el("span", { class: `dot ${state}` }),
    el("span", { text: label }),
    detail != null ? el("span", { class: "topbar-meta", text: detail }) : null,
  ]);
}
