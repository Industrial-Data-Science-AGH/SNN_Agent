#!/usr/bin/env bash
set -euo pipefail

# deploy/install.sh — idempotent one-time Pi bring-up for Wake-Up AI
#
# Run as the service user (not root); the script uses sudo for privileged steps.
# Re-running is safe: venv, .env, and systemd unit creation are guarded.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
VENV_DIR="$REPO_DIR/.venv"
VAR_DIR="${SNN_AGENT_VAR_DIR:-$HOME/.local/var/snn-agent}"
HALT_BIN="$(command -v halt 2>/dev/null || echo /usr/sbin/halt)"

echo "==> Install config"
echo "    REPO_DIR  : $REPO_DIR"
echo "    USER_NAME : $USER_NAME"
echo "    VENV_DIR  : $VENV_DIR"
echo "    VAR_DIR   : $VAR_DIR"
echo "    HALT_BIN  : $HALT_BIN"
echo ""

# ── 1/8 apt deps ────────────────────────────────────────────────────────────
echo "==> [1/8] Installing apt dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-picamera2 python3-lgpio python3-gpiozero rsync

# ── 2/8 venv ────────────────────────────────────────────────────────────────
echo "==> [2/8] Creating venv (--system-site-packages)..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt" --quiet

# ── 3/8 secrets template ────────────────────────────────────────────────────
echo "==> [3/8] Secrets template..."
mkdir -p "$HOME/.config/snn-agent"
ENV_FILE="$HOME/.config/snn-agent/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$REPO_DIR/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "    Created $ENV_FILE"
    echo "    *** Fill in GEMINI_API_KEY, GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO ***"
else
    echo "    $ENV_FILE already exists — skipping (edit manually if needed)."
fi

# ── 4/8 var dirs ────────────────────────────────────────────────────────────
echo "==> [4/8] Creating var dirs..."
mkdir -p "$VAR_DIR/clips" "$VAR_DIR/models"

# ── 5/8 group membership ────────────────────────────────────────────────────
echo "==> [5/8] Group membership (video + gpio)..."
sudo usermod -aG video,gpio "$USER_NAME"
echo "    Re-login (or reboot) required for group changes to take effect."

# ── 6/8 sudoers ─────────────────────────────────────────────────────────────
echo "==> [6/8] Rendering + installing sudoers drop-in..."
SUDOERS_SRC="$REPO_DIR/deploy/snn-agent.sudoers"
SUDOERS_RENDERED="$(mktemp)"
sed \
    -e "s|__USER__|${USER_NAME}|g" \
    -e "s|__HALT__|${HALT_BIN}|g" \
    "$SUDOERS_SRC" > "$SUDOERS_RENDERED"
sudo visudo -cf "$SUDOERS_RENDERED"
sudo cp "$SUDOERS_RENDERED" /etc/sudoers.d/snn-agent
sudo chmod 440 /etc/sudoers.d/snn-agent
rm -f "$SUDOERS_RENDERED"
echo "    Installed /etc/sudoers.d/snn-agent (NOPASSWD halt for $USER_NAME)"

# ── 7/8 systemd unit ────────────────────────────────────────────────────────
echo "==> [7/8] Rendering + installing systemd unit..."
UNIT_SRC="$REPO_DIR/deploy/snn-agent.service"
# shellcheck disable=SC2119
UNIT_RENDERED="$(mktemp --suffix=.service)"
sed \
    -e "s|__USER__|${USER_NAME}|g" \
    -e "s|__WORKDIR__|${REPO_DIR}|g" \
    -e "s|__VENV__|${VENV_DIR}|g" \
    -e "s|__VAR_DIR__|${VAR_DIR}|g" \
    "$UNIT_SRC" > "$UNIT_RENDERED"
sudo cp "$UNIT_RENDERED" /etc/systemd/system/snn-agent.service
rm -f "$UNIT_RENDERED"
sudo systemctl daemon-reload
sudo systemctl enable snn-agent.service
echo "    Enabled snn-agent.service (oneshot, fires on boot)"

# ── 8/8 next steps ──────────────────────────────────────────────────────────
echo ""
echo "==> [8/8] Installation complete."
echo ""
echo "Next steps:"
echo "  1. Fill in secrets : nano $ENV_FILE"
echo "  2. Smoke-test       : source $VENV_DIR/bin/activate && cd $REPO_DIR && python -m agent.main --test"
echo "  3. One-shot manual  : sudo systemctl start snn-agent.service"
echo "  4. Reboot to arm    : sudo reboot"
echo ""
echo "Logs   : journalctl -u snn-agent.service"
echo "Events : $VAR_DIR/event.log"
