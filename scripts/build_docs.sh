#!/bin/bash
# Build the FEMCubes Sphinx documentation.
#
# Usage:
#   bash scripts/build_docs.sh          # build HTML docs
#   bash scripts/build_docs.sh clean    # remove previous build then rebuild
#
# The generated site will be at docs/_build/html/index.html

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
PYTHON="${PYTHON:-python}"

if ! command -v "$PYTHON" &> /dev/null; then
    echo "Python was not found. Activate the FenicsX environment first."
    return 1 2>/dev/null || exit 1
fi

# Clean previous build if requested
if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning previous build..."
    rm -rf "$DOCS_DIR/_build"
fi

# Install doc dependencies if Sphinx is missing
if ! "$PYTHON" -m sphinx --version &> /dev/null; then
    echo "Sphinx not found -- installing the pinned documentation dependencies..."
    "$PYTHON" -m pip install --no-build-isolation -e "$REPO_ROOT[docs]"
fi

echo "Building HTML documentation..."
SPHINX_LOG="$(mktemp)"
trap 'rm -f "$SPHINX_LOG"' EXIT

"$PYTHON" -m sphinx -b html "$DOCS_DIR" "$DOCS_DIR/_build/html" -W --keep-going 2>&1 | tee "$SPHINX_LOG" | tail -20

echo ""
echo "Done. Open the docs with:"
echo "  xdg-open $DOCS_DIR/_build/html/index.html"
