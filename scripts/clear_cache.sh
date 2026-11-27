#!/bin/bash
# Clear cached FEniCS JIT run directories after confirmation.
#
# Usage:
#   bash scripts/clear_cache.sh
#   bash scripts/clear_cache.sh --all
#
# This removes entries matching:
#   ~/.cache/fenics/run_*
# and leaves the parent cache directory in place.

set -euo pipefail

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/fenics"
SHOW_ALL=false

if [[ "${1:-}" == "--all" ]]; then
    SHOW_ALL=true
elif [[ $# -gt 0 ]]; then
    echo "Unknown option: $1"
    echo "Usage: bash scripts/clear_cache.sh [--all]"
    exit 1
fi

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Cache directory not found: $CACHE_DIR"
    exit 0
fi

shopt -s nullglob
TARGETS=("$CACHE_DIR"/run_*)
shopt -u nullglob

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No run_* cache directories found in $CACHE_DIR"
    exit 0
fi

echo "FEniCS cache directory:"
echo "  $CACHE_DIR"
echo ""
echo "Current total size:"
du -sh "$CACHE_DIR"
echo ""
echo "Run-cache entries to be deleted: ${#TARGETS[@]}"
echo ""
if [[ "$SHOW_ALL" == true ]]; then
    echo "Entries to be deleted:"
    du -sh "${TARGETS[@]}" | sort -h
else
    echo "Largest 20 entries to be deleted:"
    du -sh "${TARGETS[@]}" | sort -h | tail -20
    if [[ ${#TARGETS[@]} -gt 20 ]]; then
        echo ""
        echo "(showing 20 of ${#TARGETS[@]} entries; use --all for the full list)"
    fi
fi
echo ""
echo "Total size to be deleted:"
du -ch "${TARGETS[@]}" | tail -1
echo ""

read -r -p "Proceed with deletion? [y/N] " reply
case "$reply" in
    y|Y|yes|YES)
        rm -rf -- "${TARGETS[@]}"
        echo ""
        echo "Deleted ${#TARGETS[@]} cache entries."
        echo "Remaining cache size:"
        du -sh "$CACHE_DIR"
        ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac
