#!/usr/bin/env bash
# Writes approx-chol's public surface to its snapshot (default, --write), or fails if they differ (--check).
set -euo pipefail

# Bumped together: cargo-public-api 0.52 reads the rustdoc JSON of nightly-2025-11-22 and later.
nightly="nightly-2026-11-01"
tool_version="0.52.0"

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
snapshot=crates/approx-chol/public-api.txt

case "${1:---write}" in
  --write) mode="write" ;;
  --check) mode="check" ;;
  *) echo "usage: scripts/public-api.sh [--write | --check]" >&2; exit 2 ;;
esac

if ! rustup toolchain list | grep "^$nightly-" >/dev/null; then
  rustup toolchain install "$nightly" --profile minimal --no-self-update
fi
if [[ "$(cargo-public-api --version 2>/dev/null)" != "cargo-public-api $tool_version" ]]; then
  cargo install --locked --force --version "$tool_version" cargo-public-api
fi

# The toolchain's own cargo first on PATH: a stable cargo there (pixi's) makes cargo-public-api fall back to plain `nightly`.
toolchain_bin=$(dirname "$(rustup which --toolchain "$nightly" cargo)")

listing=$(mktemp)
trap 'rm -f "$listing"' EXIT
# -ss drops blanket and auto-trait impls, which every type repeats; cargo-semver-checks gates auto traits.
(
  cd "$root"
  PATH="$toolchain_bin:$PATH" RUSTUP_TOOLCHAIN="$nightly" \
    cargo public-api -p approx-chol --all-features -ss --color never >"$listing"
)

if [[ $mode == write ]]; then
  cp "$listing" "$root/$snapshot"
  echo "wrote $snapshot ($(wc -l <"$listing" | tr -d ' ') lines)"
elif ! diff -u --label "$snapshot" --label "listed now" "$root/$snapshot" "$listing"; then
  echo "error: the public API differs from $snapshot (diff above)." >&2
  echo "If the change is intended, run scripts/public-api.sh and commit $snapshot." >&2
  exit 1
fi
