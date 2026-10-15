#!/usr/bin/env bash
set -euo pipefail

source <(cargo llvm-cov show-env --sh)
cargo llvm-cov clean --workspace

cargo test --workspace --all-features --locked
maturin develop --uv --locked
pytest tests/ -v

# A floor, not a target: headroom below measured coverage so one untested branch doesn't fail.
# approx-chol-next is gated once it replaces approx-chol (#182).
cargo llvm-cov report --summary-only --fail-under-lines 90 --ignore-filename-regex 'approx-chol-next/'
