#![cfg(feature = "serde")]

#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/path.rs"]
mod path;
use panic_ok::OrPanic;

use approx_chol::{factorize_with, Backend, Config, CsrRef, ExactFailure, Factor};
use rstest::rstest;

fn path_factor_with(config: Config) -> Factor<f64> {
    let row_ptrs: Vec<u32> = path::ROW_PTRS.iter().map(|&v| v as u32).collect();
    let col_indices: Vec<u32> = path::COL_INDICES.iter().map(|&v| v as u32).collect();
    let csr = CsrRef::new(&row_ptrs, &col_indices, &path::VALUES, path::N).or_panic("valid csr");
    factorize_with(csr, config).or_panic("factorization should succeed")
}

fn path_factor() -> Factor<f64> {
    path_factor_with(Config::default())
}

#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn factor_json_roundtrip_preserves_solve(#[case] backend: Backend) {
    let (row_ptrs, columns) = ([0u32, 2, 4, 6, 8], [0u32, 1, 0, 1, 2, 3, 2, 3]);
    let values = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
    let split = CsrRef::new(&row_ptrs, &columns, &values, 4).or_panic("valid CSR");

    // Strictly dominant, so ingestion grounds it and the restored factor has to
    // recover the augmentation from its blocks' anchors.
    let (sddm_row_ptrs, sddm_columns) = ([0u32, 2, 4], [0u32, 1, 0, 1]);
    let sddm_values = [2.0, -1.0, -1.0, 2.0];
    let sddm = CsrRef::new(&sddm_row_ptrs, &sddm_columns, &sddm_values, 2).or_panic("valid CSR");

    let config = Config {
        backend,
        ..Config::default()
    };
    assert_roundtrip(
        "connected path",
        &path_factor_with(config),
        &[1.0, -1.0, 1.0, -1.0],
    );
    assert_roundtrip(
        "two components",
        &factorize_with(split, config).or_panic("factorization should succeed"),
        &[1.0, -1.0, 2.0, -2.0],
    );
    let grounded = factorize_with(sddm, config).or_panic("factorization should succeed");
    assert_eq!(
        grounded.n(),
        grounded.original_n() + 1,
        "SDDM input augments by one"
    );
    assert_roundtrip("grounded SDDM", &grounded, &[1.0, -1.0]);
}

fn assert_roundtrip(label: &str, factor: &Factor<f64>, b: &[f64]) {
    let json = serde_json::to_string(factor).or_panic("serialize factor");
    let restored: Factor<f64> = serde_json::from_str(&json).or_panic("deserialize factor");

    assert_eq!(restored.n(), factor.n(), "{label}");
    assert_eq!(restored.original_n(), factor.original_n(), "{label}");
    assert_eq!(restored.n_steps(), factor.n_steps(), "{label}");
    assert_eq!(
        factor.solve(b).or_panic("solve original"),
        restored.solve(b).or_panic("solve restored"),
        "{label}: deserialized factor must reproduce the solve bit-for-bit"
    );
}

#[test]
fn deserializing_corrupted_factor_is_rejected() {
    let mut value = serde_json::to_value(path_factor()).or_panic("serialize factor");
    assert!(
        value["blocks"][0]["dim"].is_u64(),
        "no block dimension to corrupt"
    );
    value["blocks"][0]["dim"] = serde_json::Value::from(999u32);

    assert!(serde_json::from_value::<Factor<f64>>(value).is_err());
}

#[rstest]
#[case::default(Backend::default())]
#[case::approximate(Backend::Approximate)]
#[case::exact_that_errors(Backend::ExactBelow { max_dim: 8, on_failure: ExactFailure::Error })]
fn config_json_roundtrip(#[case] backend: Backend) {
    let config = Config {
        seed: 42,
        split_merge: Some(3),
        backend,
    };

    let json = serde_json::to_string(&config).or_panic("serialize config");
    let restored: Config = serde_json::from_str(&json).or_panic("deserialize config");

    assert_eq!(restored.seed, config.seed);
    assert_eq!(restored.split_merge, config.split_merge);
    assert_eq!(restored.backend, config.backend);
}

#[test]
fn config_without_a_backend_deserializes_to_the_default() {
    let restored: Config =
        serde_json::from_str(r#"{"seed":1,"split_merge":null}"#).or_panic("deserialize config");
    assert_eq!(restored.backend, Backend::default());
}
