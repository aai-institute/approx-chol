#!/usr/bin/env bash
set -euo pipefail

source <(cargo llvm-cov show-env --sh)
cargo llvm-cov clean --workspace

cargo test --workspace --all-features --locked
maturin develop --uv --locked
pytest tests/ -v

# Tracks the 90.59% both CI and a clean local run measure.
cargo llvm-cov report --summary-only --fail-under-lines 90
