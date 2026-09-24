# Edge bridge deployment (W1)

`python -m rpi_agents.agent.bridge --config edge.toml` reads the Uno serial stream, turns it into contract
`SpikeBatch` payloads, keeps them in a durable outbox and delivers them to the backend. It also polls capture
commands, drives the camera adapter and sends a `DeviceStatus` heartbeat. Standard library only, plus Pillow
for the camera (already on Raspberry Pi OS as `python3-pil`). No torch, no local vision.

What it does **not** do yet: upload images (no endpoint until W2), actuate LED/buzzer (no alarm adapter; an
alarm command is answered `failed / UNSUPPORTED_COMMAND`), or authenticate beyond a bearer token read from a
file (W4 defines the real device credential).

## Files

| File | Purpose |
|---|---|
| `edge.example.toml` | Annotated configuration; tested to stay valid |
| `snn-edge.service` | Hardened system unit for production |
| `snn-edge.user.service` | Plain user unit for development on the Pi, no sudo |

## Configure

1. Copy `edge.example.toml` to `/etc/snn-edge/edge.toml`. Set the device id, the persistent
   `/dev/serial/by-id/...` path, the channel names in firmware order, the camera USB serial and the release's
   `model_hash` / `encoder_hash`.
2. For a backend that is not on loopback: `https://` only, and a credential file
   (`chmod 600`, one token, outside any checkout). The bridge refuses a credential file that group or others can
   read, never logs the token, and never follows redirects.
3. Run once in the foreground to see configuration errors (exit code 2):
   `python -m rpi_agents.agent.bridge --config /etc/snn-edge/edge.toml`.

## Install as a system service

```sh
sudo useradd --system --home /var/lib/snn-edge --shell /usr/sbin/nologin snn-edge
sudo usermod -aG dialout,video snn-edge
sudo install -d -m 0750 -o root -g snn-edge /etc/snn-edge
sudo install -m 0640 -o root -g snn-edge edge.toml /etc/snn-edge/edge.toml
sudo install -m 0600 -o snn-edge -g snn-edge device.token /etc/snn-edge/device.token   # if used
sudo cp rpi_agents/deploy/snn-edge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now snn-edge
journalctl -u snn-edge -f
```

The unit runs unprivileged in a sandbox (no capabilities, read-only system, only the serial and video devices).
`SIGTERM` flushes the outbox for up to `limits.drain_s` and stops the session; a configuration error (exit 2) is
not restarted, any other failure is restarted after 3 s.

## Run as a user service (development)

```sh
mkdir -p ~/.config/systemd/user ~/.config/snn-edge
cp rpi_agents/deploy/snn-edge.user.service ~/.config/systemd/user/snn-edge.service
cp rpi_agents/deploy/edge.example.toml ~/.config/snn-edge/edge.toml   # then edit
systemctl --user daemon-reload && systemctl --user enable --now snn-edge
journalctl --user -u snn-edge -f
```

## Behaviour to know

| Situation | What happens |
|---|---|
| Backend unreachable, 5xx, 429, 401/403 | Retry with backoff, order kept, nothing dropped until the outbox bound |
| Backend rejects an entry for good (other 4xx) | Dead-lettered, counted in `outbox.dead`, the rest continues; the loss is a visible gap |
| Outbox or pre-session buffer full | Oldest entry dropped, counted (`outbox.dropped_total`), visible as a gap |
| Uno reset, USB reconnect or restart | New boot, so a new session; the old session is stopped first (one active session per device) |
| Frames without a boot line | The bridge asks the device to reprint it (`I`, at most once a second) |
| First bytes after opening the port | Usually the tail of an older line: ignored silently (debug log), not counted as an error |
| Boot line with another channel count or build | Refused, `state: error` with the reason; nothing is guessed |
| Port open but no valid frame | `state: stalled`, the open batch is flushed |
| Capture command | Only if a camera and an image sink exist; expires on a local monotonic clock; handled once, even across restarts |

## Limits of the W0 mock backend

The mock only accepts `localhost` / `127.0.0.1`, so for an on-device demo run it on the same Pi. It also caps
a session at 256 batches (about 64 s at 250 ms per batch, then it answers 429 and the bridge keeps retrying),
16 sessions and 1024 idempotent requests. These are mock limits, not bridge limits. It never receives images.

## Updates and rollback

Deploy a released commit, not a working directory: record the commit hash in `SNN_EDGE_VERSION` (shown in
`DeviceStatus.agent_version`), keep the previous release directory, and roll back by switching the unit's
`ExecStart` path and restarting. Do not edit code on the device; move changes through a PR.
