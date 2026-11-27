#!/bin/bash
# Serve the built FEMCubes documentation for checking it in a browser.
#
# Usage:
#   bash scripts/start_doc_server.sh                # serve docs/_build/html on port 8137
#   bash scripts/start_doc_server.sh --build        # rebuild the docs first
#   bash scripts/start_doc_server.sh --port 8200    # another port
#   bash scripts/start_doc_server.sh --host 0.0.0.0 # listen on all interfaces (other machines)
#
# The server runs in the background and keeps serving after this script ends;
# stop it with scripts/stop_doc_server.sh. It listens on localhost by default:
# over a VS Code remote connection the port is forwarded to your own machine,
# so the printed address opens there. Pages are sent without caching, so a page
# rebuilt with scripts/build_docs.sh shows on the next reload.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_DIR="$REPO_ROOT/docs/_build/html"
PID_FILE="$REPO_ROOT/docs/_build/doc_server.pid"
LOG_FILE="$REPO_ROOT/docs/_build/doc_server.log"
PYTHON="${PYTHON:-python}"
PORT=8137
HOST=127.0.0.1
BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
    esac
done

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "The documentation server is already running (process $(cat "$PID_FILE")); stop it with"
    echo "  bash scripts/stop_doc_server.sh"
    exit 1
fi

if [[ "$BUILD" == 1 || ! -f "$HTML_DIR/index.html" ]]; then
    bash "$REPO_ROOT/scripts/build_docs.sh"
fi

mkdir -p "$(dirname "$PID_FILE")"
nohup "$PYTHON" "$REPO_ROOT/scripts/doc_server.py" "$HTML_DIR" "$PORT" "$HOST" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# Report a failed start (for example a port already in use) instead of a dead address
sleep 1
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "The documentation server did not start:"
    tail -1 "$LOG_FILE"
    if grep -q "Address already in use" "$LOG_FILE"; then
        echo "Port $PORT is in use; choose another with --port."
    fi
    exit 1
fi

echo "Serving $HTML_DIR (process $(cat "$PID_FILE"))"
echo "  http://localhost:$PORT/"
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
    echo "  http://$(hostname -f):$PORT/ (from other machines)"
fi
echo "Stop it with: bash scripts/stop_doc_server.sh"
