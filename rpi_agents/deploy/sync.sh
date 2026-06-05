#!/usr/bin/env bash
set -euo pipefail

# deploy/sync.sh — rsync rpi_agents/ from Mac to Pi for fast dev iteration.
#
# Excludes .venv/ and var/ so the Pi's installed packages and saved clips
# are never overwritten by the Mac sync.
#
# Usage:
#   PI_HOST=snn-pi bash deploy/sync.sh
#   PI_HOST=192.168.1.42 bash deploy/sync.sh

PI_HOST="${PI_HOST:-snn-pi}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Syncing rpi_agents/ → ${PI_HOST}:~/SNN_Agent/rpi_agents/ ..."
rsync -avz --delete \
    --exclude='.venv/' \
    --exclude='var/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='*.pyc' \
    --exclude='tests/fixtures/images/' \
    "$REPO_DIR/" \
    "${PI_HOST}:~/SNN_Agent/rpi_agents/"

echo ""
echo "Sync complete.  Run smoke-test on Pi:"
echo "  ssh ${PI_HOST} 'cd ~/SNN_Agent/rpi_agents && source .venv/bin/activate && python -m agent.main --test'"
