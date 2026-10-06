use approx_chol::{factorize_with, Backend, Config, CsrRef, Factor, Sddm, FACTOR_FORMAT_VERSION};
use std::fs;
use std::path::Path;

#[path = "../tests/common/grid.rs"]
mod grid;

/// Written at build time rather than committed: a seed encoded before a
/// `FACTOR_FORMAT_VERSION` bump is rejected on the version check, and a corpus that rots
/// that way leaves the fuzzer reporting clean runs it never earned. Building the target is
/// the one thing nobody fuzzing can skip.
fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/factor_from_bytes");
    // Watching the corpus is what makes a pruned or deleted one come back: cargo's default
    // fingerprint ignores it, so the seeds would stay missing until something else changed.
    println!("cargo::rerun-if-changed={}", dir.display());
    println!("cargo::rerun-if-changed=build.rs");
    fs::create_dir_all(&dir).expect("create the corpus directory");
    let version = postcard::to_stdvec(&FACTOR_FORMAT_VERSION).expect("encode the version");
    for (name, factor) in seeds() {
        let bytes = postcard::to_stdvec(&factor).expect("serialize the seed factor");
        postcard::from_bytes::<Factor<f64>>(&bytes)
            .expect("a seed that does not decode starts the fuzzer outside the validator");
        // The target supplies the version itself, so a seed holds only what follows it.
        let body = bytes
            .strip_prefix(version.as_slice())
            .expect("a serialized factor opens with its format version");
        fs::write(dir.join(format!("{name}.bin")), body).expect("write the seed");
    }
}

/// One seed per shape the encoding can take, so a mutation lands in a payload that already
/// reaches the solve rather than one the framing rejects.
fn seeds() -> Vec<(&'static str, Factor<f64>)> {
    let grid = grid::grid_laplacian(4, 4);
    vec![
        ("floating_path", path(Backend::Approximate)),
        ("exact_path", path(Backend::default())),
        // Interleaved components relabel to a non-identity order, which is the only shape
        // that carries a permutation.
        (
            "permuted_components",
            factor(
                &[0, 2, 4, 6, 8],
                &[0, 2, 1, 3, 0, 2, 1, 3],
                &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                Backend::Approximate,
            ),
        ),
        // Strictly dominant, so the block is grounded and its sink is a slot past its
        // vertices instead of its own last vertex.
        (
            "grounded_sddm",
            factor(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[2.0, -1.0, -1.0, 2.0],
                Backend::Approximate,
            ),
        ),
        // Grounded components interleaved: ground slots between blocks under a permutation.
        (
            "permuted_grounded_components",
            factor(
                &[0, 2, 4, 6, 8],
                &[0, 2, 1, 3, 0, 2, 1, 3],
                &[2.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 3.0],
                Backend::Approximate,
            ),
        ),
        // Enough steps that a mutated payload can disagree about which vertex a step eliminates.
        (
            "grid_4x4",
            factor(
                &grid.row_ptrs,
                &grid.col_indices,
                &grid.values,
                Backend::Approximate,
            ),
        ),
    ]
}

/// The same matrix under both backends, so the pair differs only in the arm that factored
/// it: an elimination sequence against a packed dense factor.
fn path(backend: Backend) -> Factor<f64> {
    factor(
        &[0, 2, 5, 8, 10],
        &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        backend,
    )
}

fn factor(row_ptrs: &[u32], col_indices: &[u32], values: &[f64], backend: Backend) -> Factor<f64> {
    let n = u32::try_from(row_ptrs.len() - 1).expect("dimension fits in u32");
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid csr");
    let config = Config {
        backend,
        ..Config::default()
    };
    let sddm = Sddm::try_from(csr).expect("valid SDDM");
    factorize_with(sddm, config).expect("factorization should succeed")
}
