#!/usr/bin/env python3
"""Compare a `wallclock_banded` run against the committed reference (#60).

A regression writes a GitHub job-summary alert but never fails the build, because
a wall-clock signal on shared runners is advisory. A bench that emitted no number
is breakage rather than a slow result, so that does fail. --update refreshes the
reference instead of comparing.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_REFERENCE = "benches/wallclock_reference.json"


def emit(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")
    print(markdown)


def parse_bench_output(text: str) -> tuple[int, str]:
    best = re.findall(r"WALLCLOCK_BEST_NS=(\d+)", text)
    workload = re.findall(r"WALLCLOCK_WORKLOAD=(\S+)", text)
    if not best or not workload:
        raise SystemExit(
            "::error::bench emitted no WALLCLOCK_BEST_NS/WALLCLOCK_WORKLOAD; "
            "the harness is broken, not slow"
        )
    return int(best[-1]), workload[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument(
        "--bench-output",
        required=True,
        help="file holding the wallclock_banded harness's stdout",
    )
    p.add_argument("--commit", default="(local run)")
    p.add_argument(
        "--update",
        action="store_true",
        help="refresh the reference file if the measured time clears the noise floor, then exit",
    )
    args = p.parse_args()

    ref_path = Path(args.reference)
    reference = json.loads(ref_path.read_text())
    measured, workload = parse_bench_output(Path(args.bench_output).read_text())
    measured_ms = measured / 1e6
    baseline = reference.get("best_ns")
    baseline_workload = reference.get("workload")

    if args.update:
        min_delta_pct = reference["refresh_min_delta_pct"]
        if baseline and workload == baseline_workload:
            delta_pct = (measured - baseline) / baseline * 100.0
            if abs(delta_pct) <= min_delta_pct:
                emit(
                    f"Reference unchanged: {measured_ms:.1f} ms vs {baseline / 1e6:.1f} ms "
                    f"({delta_pct:+.1f}%) is inside the ±{min_delta_pct:.0f}% refresh floor."
                )
                return 0
        reference["workload"] = workload
        reference["best_ns"] = measured
        ref_path.write_text(json.dumps(reference, indent=2) + "\n")
        emit(f"Updated wall-clock reference to {measured_ms:.1f} ms on `{workload}`.")
        return 0

    if not baseline:
        emit(
            f"⚠️ **Perf reference not bootstrapped.** Measured {measured_ms:.1f} ms. "
            f"Run the **Perf Reference** workflow via `workflow_dispatch` to "
            f"seed it on CI hardware."
        )
        return 0

    if workload != baseline_workload:
        emit(
            f"⚠️ **Perf reference measures a different workload.** This run is "
            f"`{workload}`, the reference is `{baseline_workload}` — the two are not "
            f"comparable, so no verdict was formed. Re-seed via the **Perf "
            f"Reference** workflow."
        )
        return 0

    threshold_pct = reference["threshold_pct"]
    delta_pct = (measured - baseline) / baseline * 100.0
    baseline_ms = baseline / 1e6

    if delta_pct > threshold_pct:
        # An annotation, because a green check's job summary is a page nobody opens.
        print(
            f"::warning::wall-clock regression: {measured_ms:.1f} ms vs "
            f"{baseline_ms:.1f} ms reference (+{delta_pct:.1f}%, threshold "
            f"+{threshold_pct:.1f}%) on {args.commit}"
        )
        emit(
            f"## 🚨 Wall-clock perf regression (non-blocking)\n\n"
            f"Merge `{args.commit}` slowed `{workload}`.\n\n"
            f"| metric | value |\n|---|---|\n"
            f"| best (this merge) | {measured_ms:.1f} ms |\n"
            f"| reference | {baseline_ms:.1f} ms |\n"
            f"| change | +{delta_pct:.1f}% |\n"
            f"| threshold | +{threshold_pct:.1f}% |\n\n"
            f"If this cost is expected, refresh the reference via the "
            f"**Perf Reference** workflow."
        )
    else:
        emit(
            f"✅ Wall-clock within reference: {measured_ms:.1f} ms vs "
            f"{baseline_ms:.1f} ms ({delta_pct:+.1f}%, threshold +{threshold_pct:.1f}%)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
