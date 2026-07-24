#!/usr/bin/env bash
set -euo pipefail

cargo llvm-cov clean --workspace
source <(cargo llvm-cov show-env --sh)

cargo test --workspace --all-features
maturin develop --uv
pytest tests/ -v

cargo llvm-cov report --summary-only
