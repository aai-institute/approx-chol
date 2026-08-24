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

struct Matrix {
    name: &'static str,
    row_ptrs: &'static [u32],
    col_indices: &'static [u32],
    values: &'static [f64],
    /// `Backend::default()` is a trait method, so a const initializer cannot call it.
    backend: Option<Backend>,
}

impl Matrix {
    fn csr(&self) -> CsrRef<'_, f64, u32> {
        let n = u32::try_from(self.row_ptrs.len() - 1).expect("dimension fits in u32");
        CsrRef::new(self.row_ptrs, self.col_indices, self.values, n).expect("valid csr")
    }

    fn backend(&self) -> Backend {
        self.backend.unwrap_or_default()
    }

    /// The one place a fixture path is spelled, so the reader below and the regenerator
    /// cannot disagree about which file this build's version names.
    fn fixture_path(&self) -> String {
        format!(
            "{}/tests/fixtures/{}_{FACTOR_FORMAT_VERSION:#010x}.json",
            env!("CARGO_MANIFEST_DIR"),
            self.name
        )
    }

    fn factor(&self) -> Factor<f64> {
        let config = Config {
            backend: self.backend(),
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
};

/// Strictly dominant, so ingestion grounds it and the payload carries a ground anchor.
const GROUNDED: Matrix = Matrix {
    name: "grounded_sddm",
    row_ptrs: &[0, 2, 5, 8, 10],
    col_indices: &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    values: &[2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0],
    backend: None,
};

/// `K4` under the approximate arm: the only fixture freezing an elimination sequence, and
/// its leading column hands out shares rather than the lone `1.0` a path would pin.
const SAMPLED: Matrix = Matrix {
    name: "sampled_k4",
    row_ptrs: &[0, 4, 8, 12, 16],
    col_indices: &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    values: &[
        3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, 3.0,
    ],
    backend: Some(Backend::Approximate),
};

const FIXTURES: [&Matrix; 3] = [&INTERLEAVED, &GROUNDED, &SAMPLED];

/// Adding a fixture means adding a case below; without this the new const is regenerated
/// but never read back.
#[test]
fn every_fixture_has_a_case() {
    assert_eq!(FIXTURES.len(), 3);
}

#[rstest]
#[case::interleaved(&INTERLEAVED)]
#[case::grounded_sddm(&GROUNDED)]
#[case::sampled_k4(&SAMPLED)]
fn a_committed_payload_decodes_and_still_solves(#[case] matrix: &Matrix) {
    let committed = std::fs::read_to_string(matrix.fixture_path())
        .expect("no committed payload for this version; regenerate if the version moved");
    let restored: Factor<f64> =
        serde_json::from_str(&committed).expect("committed payload must decode");
    let fresh = matrix.factor();

    assert_eq!(restored.n(), fresh.n());
    assert_eq!(restored.original_n(), fresh.original_n());
    assert_eq!(restored.n_steps(), fresh.n_steps());

    let x = restored.solve(&B).expect("solve the restored factor");
    // A sampled factor only preconditions, so no residual bound tight enough to mean
    // anything holds; asking anyway would pass for the wrong reason.
    if !matches!(matrix.backend(), Backend::Approximate) {
        // Rows come from the matrix, not from `b`: a short right-hand side must panic in
        // the residual rather than quietly leave the last equation unjudged.
        let rows = 0..matrix.row_ptrs.len() - 1;
        let residual = residual::relative_residual_over(matrix.csr(), &x, &B, rows);
        assert!(
            residual < 1e-12,
            "the committed payload decoded to a factor that no longer solves its own \
             matrix: relative residual {residual:e}"
        );
    }

    // The residual alone is satisfied by any valid factor, not only the one that wrote these bytes.
    let expected = fresh.solve(&B).expect("solve the fresh factor");
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

/// `cargo test -p approx-chol --features serde --test wire_format -- --ignored`; the reader
/// above finds the new files by version, so only `PRE_BUMP` needs repointing.
#[test]
#[ignore = "writes fixtures; run deliberately after a format version bump"]
fn regenerate_wire_format_fixtures() {
    for matrix in FIXTURES {
        let path = matrix.fixture_path();
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
