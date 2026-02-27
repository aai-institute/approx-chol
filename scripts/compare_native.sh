#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/target/native_compare}"
RUST_OUT="$OUT_DIR/rust.csv"
JULIA_OUT="$OUT_DIR/julia.csv"
SUMMARY_OUT="$OUT_DIR/summary.csv"

for arg in "$@"; do
  if [[ "$arg" == "--out" ]]; then
    echo "--out is managed by this script. Use OUT_DIR=/path/to/dir instead." >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "[1/3] Running Rust native benchmark..."
cargo run --release -p approx-chol --example compare_native -- \
  --out "$RUST_OUT" \
  "$@"

echo "[2/3] Running Julia native benchmark..."
if command -v julia >/dev/null 2>&1; then
  JULIA_CMD=(julia)
elif command -v juliaup >/dev/null 2>&1; then
  # Ensure a default channel exists. This is idempotent if already installed.
  juliaup add release >/dev/null 2>&1 || true
  JULIA_CMD=(juliaup run release)
else
  echo "Neither 'julia' nor 'juliaup' found in PATH." >&2
  exit 1
fi

"${JULIA_CMD[@]}" "$ROOT_DIR/scripts/compare_native.jl" \
  --out "$JULIA_OUT" \
  "$@"

echo "[3/3] Aggregating results..."
python3 - "$RUST_OUT" "$JULIA_OUT" "$SUMMARY_OUT" <<'PY'
import csv
import math
import statistics
import sys
from collections import defaultdict

rust_path, julia_path, summary_path = sys.argv[1:4]

rows = []
for path in (rust_path, julia_path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(list(reader))

def p95(xs):
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = max(0, min(len(ys) - 1, math.ceil(0.95 * len(ys)) - 1))
    return ys[idx]

groups = defaultdict(list)
for r in rows:
    key = (r["impl"], r["variant"], int(r["rows"]), int(r["cols"]))
    groups[key].append(r)

summary_rows = []
for (impl_name, variant, rows_n, cols_n), group in sorted(groups.items()):
    factor_ms = [float(g["factor_ms"]) for g in group]
    apply_ms = [float(g["apply_ms"]) for g in group]
    pcg_ms = [float(g["pcg_ms"]) for g in group]
    pcg_iters = [int(g["pcg_iters"]) for g in group]
    pcg_rel = [float(g["pcg_rel_res"]) for g in group]
    precond_rel = [float(g["precond_rel_res"]) for g in group]

    summary_rows.append(
        {
            "impl": impl_name,
            "variant": variant,
            "rows": rows_n,
            "cols": cols_n,
            "cases": len(group),
            "factor_ms_median": statistics.median(factor_ms),
            "factor_ms_p95": p95(factor_ms),
            "apply_ms_median": statistics.median(apply_ms),
            "apply_ms_p95": p95(apply_ms),
            "pcg_ms_median": statistics.median(pcg_ms),
            "pcg_ms_p95": p95(pcg_ms),
            "pcg_iters_median": statistics.median(pcg_iters),
            "pcg_rel_res_median": statistics.median(pcg_rel),
            "precond_rel_res_median": statistics.median(precond_rel),
        }
    )

with open(summary_path, "w", newline="") as f:
    fieldnames = [
        "impl",
        "variant",
        "rows",
        "cols",
        "cases",
        "factor_ms_median",
        "factor_ms_p95",
        "apply_ms_median",
        "apply_ms_p95",
        "pcg_ms_median",
        "pcg_ms_p95",
        "pcg_iters_median",
        "pcg_rel_res_median",
        "precond_rel_res_median",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

def fmt_float(v, digits=3):
    return f"{v:.{digits}f}"

def fmt_sci(v):
    return f"{v:.3e}"

def render_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "-+-".join("-" * w for w in widths)
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines = [header_line, sep]
    for row in rows:
        lines.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)

# Build lookup for cross-impl comparisons.
indexed = {(r["impl"], r["variant"], r["rows"], r["cols"]): r for r in summary_rows}

# Pretty speedup table: Julia / Rust for factor and PCG.
speed_rows = []
for r in summary_rows:
    if r["impl"] != "rust":
        continue
    key_j = ("julia", r["variant"], r["rows"], r["cols"])
    j = indexed.get(key_j)
    if not j:
        continue
    factor_speedup = (
        j["factor_ms_median"] / r["factor_ms_median"]
        if r["factor_ms_median"] > 0
        else float("nan")
    )
    pcg_speedup = (
        j["pcg_ms_median"] / r["pcg_ms_median"]
        if r["pcg_ms_median"] > 0
        else float("nan")
    )
    speed_rows.append(
        [
            r["variant"],
            f"{r['rows']}x{r['cols']}",
            fmt_float(factor_speedup, 3),
            fmt_float(pcg_speedup, 3),
            fmt_float(r["pcg_iters_median"], 1),
            fmt_float(j["pcg_iters_median"], 1),
        ]
    )

print("\nSpeedup (Julia median / Rust median)")
print(
    render_table(
        ["Variant", "Size", "Factor x", "PCG x", "Iters Rust", "Iters Julia"],
        speed_rows,
    )
)

# Per-implementation median metrics table.
metric_rows = []
for r in sorted(summary_rows, key=lambda x: (x["variant"], x["rows"], x["cols"], x["impl"])):
    metric_rows.append(
        [
            r["impl"],
            r["variant"],
            f"{r['rows']}x{r['cols']}",
            fmt_float(r["factor_ms_median"], 3),
            fmt_float(r["apply_ms_median"], 3),
            fmt_float(r["pcg_ms_median"], 3),
            fmt_float(r["pcg_iters_median"], 1),
            fmt_sci(r["pcg_rel_res_median"]),
            fmt_sci(r["precond_rel_res_median"]),
        ]
    )

print("\nMedian Metrics")
print(
    render_table(
        [
            "Impl",
            "Variant",
            "Size",
            "Factor ms",
            "Apply ms",
            "PCG ms",
            "PCG iters",
            "PCG relres",
            "Precond relres",
        ],
        metric_rows,
    )
)

print(f"\nWrote summary: {summary_path}")
PY

echo
printf 'Raw outputs:\n  %s\n  %s\n  %s\n' "$RUST_OUT" "$JULIA_OUT" "$SUMMARY_OUT"
