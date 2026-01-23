#!/usr/bin/env bash
set -euo pipefail

# Start run/run_llb.py inside a detached screen session, but force "train-only":
# - do not load/run validation set
# - do not run pre-train validation
# - do not run periodic validation
#
# Usage:
#   ./run_llb_trainonly_screen_local.sh
#
# Optional env vars:
#   PYTHON=/path/to/python
#   ENTRYPOINT=run/run_llb.py
#   SCREEN_SESSION_PREFIX=llb_trainonly
#   SCREENDIR=/path/to/screen-sockets
#
# Notes:
# - This script patches configs/rl_llb_config.yaml *inside the screen session* and
#   restores it when the run finishes (so your original config/comments stay intact).

if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: screen is not installed or not on PATH."
  echo "Install it (Ubuntu/Debian): sudo apt-get update && sudo apt-get install -y screen"
  exit 1
fi

if ! tty -s; then
  echo "ERROR: no TTY detected (tty says: 'not a tty')."
  echo "Please run this script from an interactive SSH/terminal session."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/opt/conda/envs/memory/bin/python}"
ENTRY="${ENTRYPOINT:-run/run_llb.py}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python not found/executable: $PY"
  echo "Tip: set PYTHON=/path/to/python, e.g. PYTHON=\$(which python)"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$ENTRY" ]]; then
  echo "ERROR: entrypoint not found: $ROOT_DIR/$ENTRY"
  echo "Tip: set ENTRYPOINT=run/run_llb.py (or your custom entry)."
  exit 1
fi

CFG="$ROOT_DIR/configs/rl_llb_config.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG"
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
session="${SCREEN_SESSION_PREFIX:-llb_trainonly}_${ts}"
mkdir -p "$ROOT_DIR/logs"
log_file="$ROOT_DIR/logs/${session}.out"
: >"$log_file"


# Run everything inside the screen session so we can restore the config after the run ends.
cmd="$(cat <<EOF
set -euo pipefail
cfg="$CFG"
bak="\$cfg.bak_trainonly_$ts"

cp "\$cfg" "\$bak"
restore_cfg() {
  if [[ -f "\$bak" ]]; then
    mv -f "\$bak" "\$cfg"
  fi
}
trap restore_cfg EXIT

"$PY" - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

cfg = Path(r"$CFG")
txt = cfg.read_text(encoding="utf-8")

lines = txt.splitlines(keepends=True)

def find_experiment_block(ls: list[str]) -> tuple[int, int] | None:
    start = None
    for i, ln in enumerate(ls):
        if re.match(r"^experiment\s*:\s*(#.*)?$", ln):
            start = i
            break
    if start is None:
        return None
    end = len(ls)
    for j in range(start + 1, len(ls)):
        # Next top-level key (no leading spaces/tabs), ignore blank lines and comments.
        if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:\s*", ls[j]) and not ls[j].startswith((" ", "\t")):
            end = j
            break
    return start, end

blk = find_experiment_block(lines)
if blk is None:
    raise SystemExit("ERROR: could not find top-level 'experiment:' block in config")

start, end = blk
block = "".join(lines[start:end])

def set_or_insert(block: str, key: str, value: str) -> str:
    # Replace existing line (2-space indent expected in this repo's configs).
    pat = re.compile(rf"(?m)^  {re.escape(key)}\s*:\s*.*$")
    repl = f"  {key}: {value}"
    if pat.search(block):
        return pat.sub(repl, block, count=1)
    # Insert right after "experiment:" header.
    return re.sub(r"(?m)^experiment\s*:\s*(#.*)?$",
                  lambda m: m.group(0) + "\n" + repl,
                  block,
                  count=1)

# Force train-only run: no validation dataset loaded, and no validation eval scheduled.
block2 = block
block2 = set_or_insert(block2, "valid_file", "null")
block2 = set_or_insert(block2, "val_before_train", "false")
block2 = set_or_insert(block2, "valid_interval", "0")

if block2 != block:
    lines[start:end] = [block2]
    cfg.write_text("".join(lines), encoding="utf-8")
    print("[train-only] Patched configs/rl_llb_config.yaml: valid_file=null, val_before_train=false, valid_interval=0")
else:
    print("[train-only] No changes needed; config already train-only")
PY

cd "$ROOT_DIR"
"$PY" "$ENTRY" 2>&1 | tee "$log_file"
EOF
)"

screen -dmS "$session" bash -lc "$cmd"

echo "Started screen session: $session"
echo "Log file: $log_file"
echo "Attach: screen -r $session"

