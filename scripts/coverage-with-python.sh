#!/usr/bin/env bash
set -euo pipefail

source <(cargo llvm-cov show-env --sh)
cargo llvm-cov clean --workspace

cargo test --workspace --all-features --locked
maturin develop --uv --locked
pytest tests/ -v

# Floor, not a target: CI measures 93.58% lines, so this leaves room for a change to
# add an untested branch without failing on the margin alone.
cargo llvm-cov report --summary-only --fail-under-lines 90
