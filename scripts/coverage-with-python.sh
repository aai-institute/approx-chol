#!/usr/bin/env bash
set -euo pipefail

cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)

cargo test --workspace --all-features --locked
maturin develop --uv --locked
pytest tests/ -v

# Tracks the 90.59% CI measures on a clean checkout. A stale local target/
# reads higher: llvm-cov clean leaves objects from since-renamed modules.
cargo llvm-cov report --summary-only --fail-under-lines 90
