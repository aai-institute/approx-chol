use approx_chol::{factorize_with, Backend, Config, CsrRef, Factor, FACTOR_FORMAT_VERSION};
use std::fs;
use std::path::Path;

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
    let (grid_row_ptrs, grid_col_indices, grid_values) = grid(4);
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
        // Strictly dominant, so ingestion grounds it and a block anchors on the augmented
        // vertex instead of floating.
        (
            "grounded_sddm",
            factor(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &[2.0, -1.0, -1.0, 2.0],
                Backend::Approximate,
            ),
        ),
        (
            "grid_4x4",
            factor(
                &grid_row_ptrs,
                &grid_col_indices,
                &grid_values,
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
    factorize_with(csr, config).expect("factorization should succeed")
}

/// A 4-neighborhood grid Laplacian: enough elimination steps that a mutated payload can
/// disagree with itself about which vertex a step eliminates.
fn grid(side: u32) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut row_ptrs, mut col_indices, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for row in 0..side {
        for column in 0..side {
            let vertex = row * side + column;
            // Split at the diagonal rather than sorted afterwards, since a CSR row lists
            // its columns ascending.
            let mut before = Vec::new();
            let mut after = Vec::new();
            if row > 0 {
                before.push(vertex - side);
            }
            if column > 0 {
                before.push(vertex - 1);
            }
            if column + 1 < side {
                after.push(vertex + 1);
            }
            if row + 1 < side {
                after.push(vertex + side);
            }
            let degree = (before.len() + after.len()) as f64;
            col_indices.extend(before.iter().chain([&vertex]).chain(&after));
            values.extend(
                std::iter::repeat_n(-1.0, before.len())
                    .chain([degree])
                    .chain(std::iter::repeat_n(-1.0, after.len())),
            );
            row_ptrs.push(u32::try_from(col_indices.len()).expect("nnz fits in u32"));
        }
    }
    (row_ptrs, col_indices, values)
}
