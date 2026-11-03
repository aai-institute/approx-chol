# Save and load

How a caller saves $B$ and loads it in another process. The requirement is "Save and load" in
`../../requirements/consumers.md` and Persistence in `../../requirements/goals.md`.

## Decision

- **serde:** $B$ is serializable behind the `serde` feature, in the caller's format.
- **Versioned:** the saved form carries a format version, which changes only when the format
  does, and its scalar type. A load accepts only the current version and the requested scalar
  type; any other is an error naming both.
- **Validated:** a load checks every invariant of $B$, its facts included. Whatever the format
  decodes yields a valid $B$ or an error through the format's error type, never a panic.
- **Same $B$:** a loaded $B$ has the saved one's factors and facts, bit for bit; under the same
  release, its solutions too.
- **Python:** $B$ supports pickle, through the same versioned, validated form; a failed load
  raises.

## Why

- **serde:** Rust callers already hold a format.
- **Exact version:** no requirement asks to load a form across format changes, and one decoder is
  all there is to keep correct; tying the version to the format, not the release, keeps a saved
  form loadable across releases that do not touch it.
- **Scalar type recorded:** a self-describing format would otherwise cast an `f64` form into an
  `f32` $B$ silently.
- **Validated:** a saved form crosses processes and disks; a $B$ that loads must not break a solve
  or report facts its factors contradict.
- **Solutions per release:** a release may change the solve's arithmetic without touching the
  format.
- **pickle:** it is how Python saves objects and passes them to other processes, such as with
  `multiprocessing` and `joblib`; joblib's files cross releases, so they need the version too.
