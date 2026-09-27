#!/bin/sh
set -eu
# Run from the repository whose local master should be protected.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
git rev-parse --show-toplevel >/dev/null
if git config --get core.hooksPath >/dev/null; then
  echo "Existing core.hooksPath: integrate guards manually; no changes made." >&2
  exit 1
fi
hooks_dir=$(git rev-parse --git-path hooks)
for name in pre-commit pre-merge-commit pre-push; do
  if [ -e "$hooks_dir/$name" ]; then
    echo "Existing hook $name: no files overwritten; integrate manually." >&2
    exit 1
  fi
done
mkdir -p "$hooks_dir"
for name in pre-commit pre-merge-commit pre-push; do
  cp "$script_dir/hooks/$name" "$hooks_dir/$name"
  chmod +x "$hooks_dir/$name"
done
echo "Local master guards installed. GitHub rules remain authoritative."
