#![cfg(feature = "serde")]

#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/path.rs"]
mod path;
use panic_ok::OrPanic;

use approx_chol::{factorize_with, Backend, Config, CsrRef, ExactFailure, Factor};

fn path_factor() -> Factor<f64> {
    let row_ptrs: Vec<u32> = path::ROW_PTRS.iter().map(|&v| v as u32).collect();
    let col_indices: Vec<u32> = path::COL_INDICES.iter().map(|&v| v as u32).collect();
    let csr = CsrRef::new(&row_ptrs, &col_indices, &path::VALUES, path::N).or_panic("valid csr");
    factorize_with(
        csr,
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    )
    .or_panic("factorization should succeed")
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
fn dense_and_block_factors_roundtrip() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let columns = [0u32, 1, 0, 1, 2, 3, 2, 3];
    let values = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
    for config in [
        Config::default(),
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    ] {
        let csr = CsrRef::new(&row_ptrs, &columns, &values, 4).or_panic("valid CSR");
        let factor = factorize_with(csr, config).or_panic("factor");
        let encoded = serde_json::to_string(&factor).or_panic("serialize");
        let restored: Factor<f64> = serde_json::from_str(&encoded).or_panic("deserialize");
        assert_eq!(
            restored.solve(&[1.0, -1.0, 2.0, -2.0]).unwrap(),
            factor.solve(&[1.0, -1.0, 2.0, -2.0]).unwrap()
        );
    }
}

#[test]
fn deserializing_corrupted_factor_is_rejected() {
    // Asserts the serde `try_from` boundary is wired, so a structurally-corrupted
    // persisted factor is rejected at deserialize time rather than panicking on
    // solve. Per-variant coverage lives in the `factor` unit tests.
    let mut value = serde_json::to_value(path_factor()).or_panic("serialize factor");
    value["blocks"][0]["factor"]["Approx"]["sequence"]["offsets"]
        .as_array_mut()
        .expect("offsets is an array")
        .pop();

    assert!(serde_json::from_value::<Factor<f64>>(value).is_err());
}

#[test]
fn config_json_roundtrip() {
    let config = Config {
        seed: 42,
        split_merge: Some(3),
        backend: Backend::ExactBelow {
            max_dim: 24,
            on_failure: ExactFailure::FallBackToApproximate,
        },
    };

    let json = serde_json::to_string(&config).or_panic("serialize config");
    let restored: Config = serde_json::from_str(&json).or_panic("deserialize config");

    assert_eq!(restored.seed, config.seed);
    assert_eq!(restored.split_merge, config.split_merge);
    assert_eq!(restored.backend, config.backend);
}

#[test]
fn config_without_backend_uses_current_default() {
    let restored: Config =
        serde_json::from_str(r#"{"seed":42,"split_merge":2}"#).or_panic("legacy config");

    assert_eq!(restored.seed, 42);
    assert_eq!(restored.split_merge, Some(2));
    assert_eq!(restored.backend, Config::default().backend);
}

#[test]
fn dense_block_with_inconsistent_pin_is_rejected() {
    // A dense `anchor`/`ground` must equal the omitted index `m`; in-range but
    // inconsistent silently solved a different RHS in release.
    let row_ptrs = [0u32, 2, 4];
    let col_indices = [0u32, 1, 0, 1];
    let values = [2.0f64, -1.0, -1.0, 2.0];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 2).or_panic("valid csr");
    let factor = factorize_with(csr, Config::default()).or_panic("factorization");

    let json = serde_json::to_value(&factor).or_panic("serialize");
    assert_eq!(
        json["blocks"][0]["factor"]["Dense"]["m"], 2,
        "dense backend ran"
    );

    for field in ["anchor", "ground"] {
        let mut tampered = json.clone();
        tampered["blocks"][0][field] = serde_json::json!(0);
        assert!(
            serde_json::from_value::<Factor<f64>>(tampered).is_err(),
            "in-range but inconsistent `{field}` must be rejected at deserialize"
        );
    }
}
