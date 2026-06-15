#![cfg(feature = "serde")]

#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/path.rs"]
mod path;
use panic_ok::OrPanic;

use approx_chol::{factorize, Config, CsrRef, Factor};

fn path_factor() -> Factor<f64> {
    let row_ptrs: Vec<u32> = path::ROW_PTRS.iter().map(|&v| v as u32).collect();
    let col_indices: Vec<u32> = path::COL_INDICES.iter().map(|&v| v as u32).collect();
    let csr = CsrRef::new(&row_ptrs, &col_indices, &path::VALUES, path::N).or_panic("valid csr");
    factorize(csr).or_panic("factorization should succeed")
}

#[test]
fn factor_json_roundtrip_preserves_solve() {
    let factor = path_factor();

    let json = serde_json::to_string(&factor).or_panic("serialize factor");
    let restored: Factor<f64> = serde_json::from_str(&json).or_panic("deserialize factor");

    assert_eq!(restored.n(), factor.n());
    assert_eq!(restored.original_n(), factor.original_n());
    assert_eq!(restored.n_steps(), factor.n_steps());

    // RHS must lie in the range of the Laplacian (sum to zero).
    let b = [1.0, -1.0, 1.0, -1.0];
    let x_orig = factor.solve(&b).or_panic("solve original");
    let x_restored = restored.solve(&b).or_panic("solve restored");
    assert_eq!(
        x_orig, x_restored,
        "deserialized factor must reproduce the solve bit-for-bit"
    );
}

#[test]
fn config_json_roundtrip() {
    let config = Config {
        seed: 42,
        split_merge: Some(3),
    };

    let json = serde_json::to_string(&config).or_panic("serialize config");
    let restored: Config = serde_json::from_str(&json).or_panic("deserialize config");

    assert_eq!(restored.seed, config.seed);
    assert_eq!(restored.split_merge, config.split_merge);
}
