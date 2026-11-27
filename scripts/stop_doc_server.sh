#!/bin/bash
# Stop the documentation server started with scripts/start_doc_server.sh.
#
# Usage:
#   bash scripts/stop_doc_server.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$REPO_ROOT/docs/_build/doc_server.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No documentation server is running (no $PID_FILE)."
    exit 0
fi

PID="$(cat "$PID_FILE")"
# Stop only the process this repository's start script recorded, and only if it is the docs server
if kill -0 "$PID" 2>/dev/null && ps -p "$PID" -o args= | grep -q "doc_server.py"; then
    kill "$PID"
    echo "Stopped the documentation server (process $PID)."
else
    echo "The recorded server (process $PID) is no longer running."
fi
rm -f "$PID_FILE"
