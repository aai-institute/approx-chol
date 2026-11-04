#!/usr/bin/env bash
set -euo pipefail

# The wrapper instruments behind Cargo's fingerprints: a plain run would reuse what it built, so it gets its own target dir.
# `maturin develop` installs into the source tree, so the extension is rebuilt plain once the instrumented one is done.
trap 'maturin develop --uv --locked' EXIT

(
  export CARGO_TARGET_DIR="$PWD/target/llvm-cov-target"
  source <(cargo llvm-cov show-env --sh)
  cargo llvm-cov clean --workspace

  cargo test --workspace --all-features --locked --no-fail-fast
  maturin develop --uv --locked
  pytest tests/ -v

  # A floor, not a target: headroom below measured coverage so one untested branch doesn't fail.
  cargo llvm-cov report --summary-only --fail-under-lines 90
)
