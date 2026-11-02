# Build

How a caller turns an admissible matrix into the approximate inverse $B$. The requirements are
"Build" and "Build for a Schur complement" in `../../requirements/consumers.md`.

## Decision

- **Arguments:** an `Input`; a configuration (fill-in level, seed, scaling tolerance and step
  limit, largest component factored exactly), valid by construction, each setting defaulted; a
  thread budget (a nonzero count or all cores), named in Rust, all cores by default in Python;
  an observer, `()` for none, which receives diagnostics and may stop the build.
- **Result:** $B$, stopped, or a `BuildError` when a computed scale, scaled entry or row sum leaves
  the representable range, located in the caller's numbering.
- **Facts on $B$**, of the target $A_r$ in the caller's numbering: the components, and per
  component its null vector if singular and its clamp if past rounding. A clamp states whether
  the scaling settled: settled proves the component is not GDD; unsettled means the steps ran out.
  Clamped rows, step counts, fill and durations go only to the observer.
- **Schur complement:** the same, with a `BipartiteFactorWidth2` and the side to keep; $B$
  approximates the pseudoinverse of the Schur complement of $A_r$ on that side, and its facts are
  numbered there.
- **Threads:** the caller's pool, never a new one, at most the budget.

## Why

- **`BuildError`:** Submit decides everything the entries decide; the scaling's range depends on
  the computed scaling, so only the build can find it.
- **Budget apart from the configuration:** it does not change $B$, so the configuration stays a
  value that can be compared and saved.
- **Budget named in Rust:** a Rust caller often runs its own parallelism; no default fits inside and
  outside a parallel loop. A Python caller expects the machine.
- **Stop as an outcome:** the caller asked for it.
- **Facts on $B$:** the solve needs the null vectors itself, and they persist with $B$.
- **Diagnostics through the observer:** they serve watching, profiling and tests, not the solve; a
  caller who does not observe pays nothing.
