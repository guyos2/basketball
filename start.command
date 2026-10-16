#!/bin/bash
# Mac / Linux: double-click (Mac) or run "bash start.command". First run installs everything.
cd "$(dirname "$0")" || exit 1

fail() { echo; echo "$1"; echo; read -r -p "Press Enter to close..."; exit 1; }

PY=""
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.10+ was not found. Install Python 3.12 from https://www.python.org/downloads/ and run this again."

if [ ! -x .venv/bin/python ]; then
  echo "Creating the environment with $PY ..."
  "$PY" -m venv .venv || fail "Could not create the environment (.venv)."
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing - the first run takes 5-10 minutes..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt || fail "Installation failed - see the messages above."
  cp requirements.txt .venv/requirements.installed
fi

.venv/bin/python -m courtvision serve --open
