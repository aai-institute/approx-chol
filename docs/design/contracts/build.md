# Build

How a caller turns an admissible matrix into the approximate inverse $B$. The requirements are
"Build" and "Build for a Schur complement" in `../../requirements/consumers.md`.

## Decision

- **Arguments:** an `Input`; a configuration (clique sample rule and fill-in level, seed,
  scaling tolerance and step limit, largest component factored exactly, how a frustrated component
  is factored, how a Schur complement is formed), valid by
  construction, each setting defaulted, serializable behind the `serde` feature; a thread budget
  (a nonzero count or all cores), all cores in the Rust one-call functions and by default in
  Python, named at the two-stage level.
- **Result:** $B$ or a `BuildError`, located in the caller's numbering: a computed
  scale, scaled entry or scaled row sum leaves the representable range, or a component has more
  than $1/\varepsilon$ vertices.
- **Facts on $B$**, of the target $A_r$ in the caller's numbering: the components, and per
  component its null vector if singular and its clamp if past the rounding band. A clamp states
  whether the scaling was refuted, proving the component is not GDD, or unsettled, its steps run
  out.
- **Schur complement:** the same, with a `BipartiteFactorWidth2` and the side to keep; $B$
  approximates the pseudoinverse of the Schur complement of $A_r$ on that side; components and
  null vectors are the complement's, numbered there, and clamps are per component of $A$.
- **Threads:** as Execution, `../architecture.md`. $B$ is bit-identical for every budget.

## Why

- **`BuildError`:** both cases need the build's own computation, the scaling or the components;
  Submit checks what single entries and rows decide.
- **Budget apart from the configuration:** it does not change $B$, so the configuration stays a
  value that can be compared and saved.
- **Two levels:** one call for the common case, with defaults; two stages, reduce then factor, for
  every setting and for a caller who reads the clamps before paying for the factorization.
- **Budget named at the two-stage level:** a caller inside its own parallel loop needs that level
  anyway; one call, like Python, expects the machine.
- **Facts on $B$:** the solve needs the null vectors itself, and they persist with $B$.
- **Serializable configuration:** Reproducibility promises the same $B$ from the same input,
  configuration and seed; a caller records the configuration to rebuild it.
