#!/bin/bash
# Re-install femcubes in editable mode.
#
# Usage:
#   bash scripts/reinstall.sh
#
# scikit-build-core automatically compiles the C++ extension during install,
# so no separate cmake invocation is needed.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Installing femcubes in editable mode (includes C++ build) ==="
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE_PACKAGES}/femcubes"
python -m pip install --no-build-isolation -e "$REPO_ROOT"

echo "=== Done ==="
