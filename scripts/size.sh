#!/usr/bin/env bash
# Lines of src code, src docs and tests at BASE and HEAD; a refactor fails if src code or tests grow.
# HEAD "." reads the working tree. KIND defaults to the branch's prefix, refactor/… → refactor.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
head=${2:-.}
base=${1:-$(git merge-base origin/main HEAD)}
branch=$(git rev-parse --abbrev-ref HEAD)
kind=${3:-${branch%%/*}}

files() { # rev extension-regex dir… → their files' contents
  local rev=$1 ext=$2; shift 2
  if [[ $rev == . ]]; then
    git ls-files -co --exclude-standard -- "$@" | grep -E "$ext" | while read -r f; do [[ -f $f ]] && cat "$f"; done
  else
    git ls-tree -r --name-only "$rev" -- "$@" | grep -E "$ext" | while read -r f; do git show "$rev:$f"; done
  fi
}

# Non-blank lines of src as "code docs tests"; a #[cfg(test)] item counts as tests up to its closing brace.
src() {
  files "$1" '\.rs$' crates/approx-chol/src | awk '
    /^[ \t]*$/ { next }
    skip { tests++; o = gsub(/\{/, "{"); c = gsub(/\}/, "}"); d += o - c; if (o) s = 1; if (s && d <= 0) skip = 0; next }
    /^[ \t]*#\[cfg\(test\)\]/ { skip = 1; d = 0; s = 0; tests++; next }
    /^[ \t]*\/\/[\/!]/ { docs++; next }
    /^[ \t]*\/\// { next }
    { code++ }
    END { print code + 0, docs + 0, tests + 0 }'
}

tests() { files "$1" '\.(rs|py)$' crates/approx-chol/tests tests | grep -cv '^[[:space:]]*$' || true; }

read -r bc bd bt < <(src "$base")
read -r hc hd ht < <(src "$head")
bt=$((bt + $(tests "$base")))
ht=$((ht + $(tests "$head")))

printf '%-9s %7s %7s %7s\n' "" base head delta
printf '%-9s %7d %7d %+7d\n' "src code" "$bc" "$hc" $((hc - bc)) "src docs" "$bd" "$hd" $((hd - bd)) tests "$bt" "$ht" $((ht - bt))

if [[ $kind == refactor ]] && ((hc > bc || ht > bt)); then
  echo "a refactor must not grow src code or tests; if this one must, label it feat or perf and say why" >&2
  exit 1
fi
