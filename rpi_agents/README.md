# rpi_agents — Wake-Up AI RPi 5 Agent

Rule-based state machine that wakes on an SNN hardware trigger, runs a
local prefilter + Gemini vision check, and fires actuators + email on a
confirmed intrusion.

State flow: `WAKE → CAPTURE → PREFILTER → [VISION] → DECIDE → re-halt`

See [`docs/docs_index.md`](docs/docs_index.md) for the full analysis and
[`../docs/`](../docs/) for hardware design.

---

## Mac dev setup (eval harness + unit tests only)

```bash
python -m venv .venv
source .venv/bin/activate

# picamera2 / gpiozero / lgpio install ONLY on the Pi (via apt).
# On Mac, install only the packages needed for the eval harness:
pip install numpy pytest python-dotenv google-genai opencv-python-headless

cd rpi_agents
python -m pytest tests/ -q
```

## Run the import-safety smoke test

```bash
cd rpi_agents
python -m agent.main --test
```

## USB camera for first tests (before the CSI module is wired)

The default backend is `csi` (picamera2).  To use a USB webcam during
bring-up, opt in with env vars — no code change needed:

```bash
ls /dev/video*                        # confirm the camera enumerated
export SNN_AGENT_CAMERA_BACKEND=usb   # default is 'csi'
export SNN_AGENT_CAMERA_INDEX=0       # /dev/video0
SNN_AGENT_POWER_MODE=warm python -m agent.main
```

Requirements: the service user must be in the `video` group (the P5
`install.sh` adds it — re-login required after first install).

To switch back to the CSI camera: `unset SNN_AGENT_CAMERA_BACKEND`.
USB is a bring-up/test affordance; CSI remains the default and the
deployed configuration.

## Pi setup

```bash
sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy secrets template and fill in values:
mkdir -p ~/.config/snn-agent
cp .env.example ~/.config/snn-agent/.env
chmod 600 ~/.config/snn-agent/.env
```

## Secrets

Secrets live in `~/.config/snn-agent/.env` on the Pi — never in the repo.
See [`.env.example`](.env.example) for the required keys.

## Architecture reference

- [`/Users/vityk/Documents/Knowledgebase/Projects/SNN_Agent/architecture.md`](../docs/project_analysis.md)
- [`docs/docs_index.md`](docs/docs_index.md)

---

## Deploy (on the Pi)

The `deploy/` directory contains everything needed to run the agent as a
`systemd` oneshot service that fires on every boot, runs one wake cycle, then
re-halts the Pi.

### One-time setup

Run on the Pi (as your normal user, not root):

```bash
cd ~/SNN_Agent/rpi_agents
bash deploy/install.sh
```

`install.sh` is idempotent — safe to re-run.  It:

1. `apt-get install` picamera2, lgpio, gpiozero, rsync
2. Creates `.venv --system-site-packages` and `pip install -r requirements.txt`
3. Copies `.env.example` → `~/.config/snn-agent/.env` (skips if present)
4. Creates `$SNN_AGENT_VAR_DIR/clips` and `models`
5. Adds the service user to `video` + `gpio` groups (re-login required)
6. Renders and validates `deploy/snn-agent.sudoers` → `/etc/sudoers.d/snn-agent`
   (`visudo -cf` runs before activation; see [POWER CONTRACT](#power-contract) below)
7. Renders and installs `deploy/snn-agent.service` → `/etc/systemd/system/`; enables it

Then fill in secrets:

```bash
nano ~/.config/snn-agent/.env   # GEMINI_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO
```

Smoke-test before rebooting:

```bash
source .venv/bin/activate
python -m agent.main --test
```

Trigger one production cycle manually (optional):

```bash
sudo systemctl start snn-agent.service
journalctl -u snn-agent.service -f
```

Arm on boot:

```bash
sudo reboot
```

### Dev loop (Mac → Pi)

```bash
# From the Mac, inside rpi_agents/:
PI_HOST=snn-pi bash deploy/sync.sh
```

`sync.sh` rsyncs `rpi_agents/` to the Pi, excluding `.venv/` and `var/` so
installed packages and saved clips are never overwritten.  Then smoke-test:

```bash
ssh snn-pi 'cd ~/SNN_Agent/rpi_agents && source .venv/bin/activate && python -m agent.main --test'
```

### Logs and events

```bash
journalctl -u snn-agent.service        # systemd journal (current boot)
journalctl -u snn-agent.service -b -1  # previous boot
cat $SNN_AGENT_VAR_DIR/event.log       # structured event log
```

### POWER CONTRACT

`agent/power.py:resleep()` calls `sudo halt` as the non-root service user.
`deploy/snn-agent.sudoers` grants `NOPASSWD` for the resolved `halt` binary
(Bookworm: `/usr/sbin/halt`).  Without it the Pi never re-halts after a wake
cycle and the system fails to re-arm.

### Network-online latency trade-off

The unit declares `After=network-online.target` so email and Gemini are
available on boot.  The local LED+buzzer alarm path does not need the network,
so if cold-boot latency is critical, you may remove `After=`/`Wants=` from the
rendered unit — but email notifications and Gemini vision will fail until the
network is ready.  For the dev/`warm` mode, set `SNN_AGENT_POWER_MODE=warm`
(default) in the environment; the unit hard-codes `halt` for production.

---

## On-hardware E2E runbook (P5 acceptance checklist)

Run manually on the Pi after `install.sh` and a successful smoke-test.

| # | Step | Expected result |
|---|------|-----------------|
| 1 | From halt: pull GPIO3 low via Arduino/jumper | Pi boots; `journalctl -u snn-agent` shows `WAKE` log |
| 2 | `WAKE_CONFIRM_PIN` read | Log shows trigger confirmed (or times out gracefully) |
| 3 | Capture → prefilter → decide runs end-to-end | State machine completes without exception |
| 4 | **False path**: static scene, no person | Clip saved to `$VAR_DIR/clips/`; no alarm; Pi re-halts |
| 5 | **Real path**: wave hand / person in frame | Gemini called; ALARM state; LED + buzzer; email w/ snapshot sent; clip saved; Pi re-halts |
| 6 | **Fail-safe**: disconnect network before step 5 | Gemini timeout → defaults to ALARM (never silent miss) |
| 7 | Measure wake-to-alarm latency (p95) | ≤ 35 s in `halt` mode; ≤ 3 s in `warm` mode |
| 8 | Confirm Pi returns to low-power halt | `systemctl status` shows inactive; board draws idle current |
