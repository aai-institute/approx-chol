#![cfg(feature = "serde")]
//! Frozen payloads from earlier builds; `serde_roundtrip.rs` writes and reads with one
//! build, so an encoding shift is invisible to it and visible here.

#[path = "common/residual.rs"]
mod residual;

use approx_chol::{factorize_with, Backend, Config, CsrRef, Factor, FACTOR_FORMAT_VERSION};
use rstest::rstest;

/// The interleaved payload as it was written before the version moved to `0x41430004`.
const PRE_BUMP: &str = include_str!("fixtures/pre_bump_0x41430003.json");
const PRE_BUMP_VERSION: u32 = 0x4143_0003;

/// Zero-sum over each component, so the floating case has an exact solution.
const B: [f64; 4] = [1.0, 2.0, -1.0, -2.0];

/// The same, for a five-vertex fixture: a short right-hand side would be zero-extended and
/// leave the last row's equation out of the residual.
const B5: [f64; 5] = [1.0, 2.0, -1.0, -2.0, 0.0];

struct Matrix {
    name: &'static str,
    row_ptrs: &'static [u32],
    col_indices: &'static [u32],
    values: &'static [f64],
    /// `None` takes the crate default rather than restating it here.
    backend: Option<Backend>,
    /// An exactly-factored fixture solves its own matrix; a sampled one only preconditions
    /// it, so what pins a sampled payload is its agreement with a fresh factor below.
    residual_bound: f64,
    /// Sized to this matrix, so every row reaches the residual.
    rhs: &'static [f64],
}

impl Matrix {
    fn csr(&self) -> CsrRef<'_, f64, u32> {
        let n = u32::try_from(self.row_ptrs.len() - 1).expect("dimension fits in u32");
        CsrRef::new(self.row_ptrs, self.col_indices, self.values, n).expect("valid csr")
    }

    fn factor(&self) -> Factor<f64> {
        let config = Config {
            backend: self.backend.unwrap_or_default(),
            ..Config::default()
        };
        factorize_with(self.csr(), config).expect("factorization should succeed")
    }
}

/// Contiguous components relabel to the identity, which `Permutation::from_order` drops.
const INTERLEAVED: Matrix = Matrix {
    name: "interleaved",
    row_ptrs: &[0, 2, 4, 6, 8],
    col_indices: &[0, 2, 1, 3, 0, 2, 1, 3],
    values: &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    backend: None,
    residual_bound: 1e-12,
    rhs: &B,
};

/// Strictly dominant, so ingestion grounds it and the payload carries a ground anchor.
const GROUNDED: Matrix = Matrix {
    name: "grounded_sddm",
    row_ptrs: &[0, 2, 5, 8, 10],
    col_indices: &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    values: &[2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0],
    backend: None,
    residual_bound: 1e-12,
    rhs: &B,
};

/// `K5` under the approximate arm: the default backend takes a block this small exactly,
/// so the encoding of an elimination sequence would otherwise go unfrozen — and every step
/// here carries several neighbors, so the composed coefficients are something other than
/// the lone `1.0` a two-vertex block would pin.
const SAMPLED: Matrix = Matrix {
    name: "sampled_k5",
    row_ptrs: &[0, 5, 10, 15, 20, 25],
    col_indices: &[
        0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
    ],
    values: &[
        4.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0, 4.0,
    ],
    backend: Some(Backend::Approximate),
    residual_bound: 0.5,
    rhs: &B5,
};

const FIXTURES: [&Matrix; 3] = [&INTERLEAVED, &GROUNDED, &SAMPLED];

#[rstest]
#[case::interleaved(&INTERLEAVED, include_str!("fixtures/interleaved_0x41430004.json"))]
#[case::grounded_sddm(&GROUNDED, include_str!("fixtures/grounded_sddm_0x41430004.json"))]
#[case::sampled_k5(&SAMPLED, include_str!("fixtures/sampled_k5_0x41430004.json"))]
fn a_committed_payload_decodes_and_still_solves(#[case] matrix: &Matrix, #[case] committed: &str) {
    let restored: Factor<f64> = serde_json::from_str(committed)
        .expect("committed payload must decode; regenerate it if the format version moved");
    let fresh = matrix.factor();

    assert_eq!(restored.n(), fresh.n());
    assert_eq!(restored.original_n(), fresh.original_n());
    assert_eq!(restored.n_steps(), fresh.n_steps());

    let b = matrix.rhs;
    assert_eq!(
        b.len(),
        matrix.row_ptrs.len() - 1,
        "the fixture's right-hand side must cover every row, or the residual skips one"
    );
    let x = restored.solve(b).expect("solve the restored factor");
    let residual = residual::relative_residual_over(matrix.csr(), &x, b, 0..b.len());
    assert!(
        residual < matrix.residual_bound,
        "the committed payload decoded to a factor that no longer solves its own matrix: \
         relative residual {residual:e} against a bound of {:e}",
        matrix.residual_bound
    );

    // The residual alone is satisfied by any valid factor, not only the one that wrote these bytes.
    let expected = fresh.solve(b).expect("solve the fresh factor");
    assert!(
        x.iter()
            .zip(&expected)
            .all(|(got, want)| (got - want).abs() < 1e-12),
        "the committed payload solves differently from this build: {x:?} against {expected:?}"
    );
}

#[test]
fn a_payload_from_before_the_last_bump_is_rejected_by_its_version() {
    let error = serde_json::from_str::<Factor<f64>>(PRE_BUMP)
        .expect_err("a pre-bump payload must not decode")
        .to_string();

    assert!(
        error.contains(&format!("{PRE_BUMP_VERSION:#010x}")),
        "error must name the version it found, got: {error}"
    );
    assert!(
        error.contains(&format!("{FACTOR_FORMAT_VERSION:#010x}")),
        "error must name the version this build reads, got: {error}"
    );
}

/// `cargo test -p approx-chol --features serde --test wire_format -- --ignored`, then point
/// each `include_str!` at the new file and `PRE_BUMP` at what it replaced.
#[test]
#[ignore = "writes fixtures; run deliberately after a format version bump"]
fn regenerate_wire_format_fixtures() {
    for matrix in FIXTURES {
        let path = format!(
            "{}/tests/fixtures/{}_{FACTOR_FORMAT_VERSION:#010x}.json",
            env!("CARGO_MANIFEST_DIR"),
            matrix.name
        );
        // Rewriting a committed payload with today's encoder is how this test comes to agree
        // with the drift it exists to catch.
        assert!(
            !std::path::Path::new(&path).exists(),
            "{path} already exists; delete it first if you really mean to unfreeze it"
        );
        let json =
            serde_json::to_string_pretty(&matrix.factor()).expect("serialize the fixture factor");
        std::fs::write(&path, format!("{json}\n")).expect("write the fixture");
        println!("wrote {path}");
    }
}
