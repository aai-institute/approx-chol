#![cfg(feature = "serde")]
//! Payloads written by earlier builds, read back by this one. `serde_roundtrip.rs`
//! writes and reads with the same build, so it cannot see an encoding shift; these
//! bytes are frozen and it can.

#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/residual.rs"]
mod residual;
use panic_ok::OrPanic;

use approx_chol::{factorize_with, Config, CsrRef, Factor, FACTOR_FORMAT_VERSION};
use rstest::rstest;

/// Written by `a06782e^`, the last build before the version moved to `0x41430002`.
const PRE_BUMP: &str = include_str!("fixtures/pre_bump_0x41430001.json");
const PRE_BUMP_VERSION: u32 = 0x4143_0001;

/// Zero-sum over each component, so the floating case has an exact solution.
const B: [f64; 4] = [1.0, 2.0, -1.0, -2.0];

struct Matrix {
    row_ptrs: &'static [u32],
    col_indices: &'static [u32],
    values: &'static [f64],
}

impl Matrix {
    fn csr(&self) -> CsrRef<'_, f64, u32> {
        CsrRef::new(self.row_ptrs, self.col_indices, self.values, 4).or_panic("valid csr")
    }

    fn factor(&self) -> Factor<f64> {
        factorize_with(self.csr(), Config::default()).or_panic("factorization should succeed")
    }
}

/// Components `{0,2}` and `{1,3}`: interleaved, so the payload carries more than one
/// block and a non-identity permutation. Contiguous components relabel to the identity,
/// which `Permutation::from_order` returns as `None`.
const INTERLEAVED: Matrix = Matrix {
    row_ptrs: &[0, 2, 4, 6, 8],
    col_indices: &[0, 2, 1, 3, 0, 2, 1, 3],
    values: &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
};

/// Strictly dominant, so ingestion grounds it and the payload carries a ground anchor.
const GROUNDED: Matrix = Matrix {
    row_ptrs: &[0, 2, 5, 8, 10],
    col_indices: &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    values: &[2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0],
};

const FIXTURES: [(&str, &Matrix); 2] =
    [("interleaved", &INTERLEAVED), ("grounded_sddm", &GROUNDED)];

#[rstest]
#[case::interleaved(&INTERLEAVED, include_str!("fixtures/interleaved_0x41430002.json"))]
#[case::grounded_sddm(&GROUNDED, include_str!("fixtures/grounded_sddm_0x41430002.json"))]
fn a_committed_payload_decodes_and_still_solves(#[case] matrix: &Matrix, #[case] committed: &str) {
    let restored: Factor<f64> =
        serde_json::from_str(committed).or_panic("the committed payload must decode");
    let fresh = matrix.factor();

    assert_eq!(restored.n(), fresh.n());
    assert_eq!(restored.original_n(), fresh.original_n());
    assert_eq!(restored.n_steps(), fresh.n_steps());

    let x = restored.solve(&B).or_panic("solve the restored factor");
    let residual = residual::relative_residual_over(matrix.csr(), &x, &B, 0..B.len());
    assert!(
        residual < 1e-12,
        "the committed payload decoded to a factor that no longer solves its own matrix: \
         relative residual {residual:e}"
    );
}

#[test]
fn a_payload_from_before_the_last_bump_is_rejected_by_its_version() {
    let error = serde_json::from_str::<Factor<f64>>(PRE_BUMP)
        .err()
        .or_panic("a pre-bump payload must not decode")
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

/// Refreshing fixtures is deliberate: run
/// `cargo test -p approx-chol --features serde --test wire_format -- --ignored`,
/// then point each `include_str!` at the new file and `PRE_BUMP` at what it replaced.
#[test]
#[ignore = "writes fixtures; run deliberately after a format version bump"]
fn regenerate_wire_format_fixtures() {
    for (name, matrix) in FIXTURES {
        let json =
            serde_json::to_string_pretty(&matrix.factor()).or_panic("serialize the fixture factor");
        let path = format!(
            "{}/tests/fixtures/{name}_{FACTOR_FORMAT_VERSION:#010x}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::write(&path, format!("{json}\n")).or_panic("write the fixture");
        println!("wrote {path}");
    }
}
