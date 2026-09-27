// Operator sign-in against Wiktor's API contract:
//   POST /auth/login    { username, password }  -> sets httponly session cookie
//   GET  /auth/session                          -> whoami
//   POST /auth/logout    (X-CSRF-Token header)   -> clears the session
//
// Security notes (task C1):
//  - The session lives in an httponly cookie the browser sends automatically;
//    JS never reads or stores it.
//  - The password is only ever the value typed into the form; it is passed to
//    fetch and never kept, never written to localStorage/sessionStorage.
//  - The CSRF token is held in memory here and cleared on logout.

import { DataError } from "./data.js";

let _csrfToken = null; // in-memory only

export function csrfToken() {
  return _csrfToken;
}

async function postJSON(url, body, headers = {}) {
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...headers },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res;
}

/** Returns { actor, expiresAt } on success; throws DataError on failure. */
export async function login(username, password) {
  let res;
  try {
    res = await postJSON("/auth/login", { username, password });
  } catch {
    throw new DataError("NETWORK", "Could not reach the server", 0);
  }
  if (res.status === 401) throw new DataError("INVALID_CREDENTIALS", "Invalid credentials", 401);
  if (res.status === 429) {
    const retry = res.headers.get("Retry-After");
    throw new DataError("TOO_MANY_ATTEMPTS",
      retry ? `Too many attempts. Try again in ${retry}s.` : "Too many attempts. Try again later.", 429);
  }
  if (!res.ok) throw new DataError("ERROR", `Sign-in failed (${res.status})`, res.status);

  const body = await res.json();
  _csrfToken = body.csrf_token || null;
  return { actor: body.actor, expiresAt: body.expires_at };
}

/** Returns the session info if a valid cookie exists, else null. */
export async function currentSession() {
  let res;
  try {
    res = await fetch("/auth/session", { credentials: "same-origin" });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const body = await res.json();
  _csrfToken = body.csrf_token || null;
  return { actor: body.actor, expiresAt: body.expires_at };
}

export async function logout() {
  try {
    await postJSON("/auth/logout", null, _csrfToken ? { "X-CSRF-Token": _csrfToken } : {});
  } catch {
    // Best effort: even if the network call fails, drop the local token.
  }
  _csrfToken = null;
}
