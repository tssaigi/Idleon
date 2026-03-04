#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
cd "$SCRIPT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi

REQ_HASH=$(shasum requirements.txt | awk '{print $1}')
STAMP_FILE=".venv/.requirements_hash"

if [[ ! -f "$STAMP_FILE" ]] || [[ "$(cat "$STAMP_FILE" 2>/dev/null)" != "$REQ_HASH" ]]; then
  .venv/bin/pip install -r requirements.txt
  printf "%s\n" "$REQ_HASH" > "$STAMP_FILE"
fi

exec .venv/bin/python -m gaming_idleon
