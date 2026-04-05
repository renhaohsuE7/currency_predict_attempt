#!/usr/bin/env bash
# Clean pipeline output. Downloaded data is preserved by default.
#
# Usage:
#   ./scripts/clean.sh              # remove results/ and models/ only
#   ./scripts/clean.sh --all        # also remove downloaded data (data/)
#
# Docker:
#   docker compose run --rm prod bash scripts/clean.sh
#   docker compose run --rm prod bash scripts/clean.sh --all

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

remove_all_data=false
for arg in "$@"; do
    case "$arg" in
        --all) remove_all_data=true ;;
        -h|--help)
            echo "Usage: $0 [--all]"
            echo "  --all    Also remove downloaded data (data/)"
            echo "  Default: only remove results/ and models/"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "Cleaning pipeline output..."

# results/ — runs, latest symlink, figures, reports
if [ -d "$PROJECT_DIR/results" ]; then
    rm -rf "$PROJECT_DIR/results/runs" \
           "$PROJECT_DIR/results/latest" \
           "$PROJECT_DIR/results/figures" \
           "$PROJECT_DIR/results/comparison_report.md" \
           "$PROJECT_DIR/results/transformer_training" \
           "$PROJECT_DIR/results/lightning_training" \
           "$PROJECT_DIR/results/models"
    # Remove any leftover files at results root (pipeline_results.json, etc.)
    find "$PROJECT_DIR/results" -maxdepth 1 -type f -delete 2>/dev/null || true
    echo "  results/ cleaned"
fi

# models/ — trained model artifacts
if [ -d "$PROJECT_DIR/models" ]; then
    rm -rf "$PROJECT_DIR/models"/*
    echo "  models/ cleaned"
fi

# data/ — downloaded Yahoo Finance data (only with --all)
if [ "$remove_all_data" = true ]; then
    if [ -d "$PROJECT_DIR/data" ]; then
        rm -rf "$PROJECT_DIR/data"/*
        echo "  data/ cleaned (downloaded data removed)"
    fi
else
    echo "  data/ preserved (use --all to remove)"
fi

echo "Done."
