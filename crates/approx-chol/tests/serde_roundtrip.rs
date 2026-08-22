#![cfg(feature = "serde")]

#[path = "common/path.rs"]
mod path;

use approx_chol::{
    factorize_with, Backend, Config, CsrRef, ExactFailure, Factor, FACTOR_FORMAT_VERSION,
};
use rstest::rstest;

fn path_factor_with(config: Config) -> Factor<f64> {
    let row_ptrs: Vec<u32> = path::ROW_PTRS.iter().map(|&v| v as u32).collect();
    let col_indices: Vec<u32> = path::COL_INDICES.iter().map(|&v| v as u32).collect();
    let csr = CsrRef::new(&row_ptrs, &col_indices, &path::VALUES, path::N).expect("valid csr");
    factorize_with(csr, config).expect("factorization should succeed")
}

fn path_factor() -> Factor<f64> {
    path_factor_with(Config::default())
}

/// Varied weights, so no two neighbors take the same share and a column of `n - 1` of them
/// is something a path or a `K4` cannot stand in for.
fn complete_laplacian(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let weight = |i: usize, j: usize| 1.0 + ((i.min(j) * 7 + i.max(j) * 3) % 11) as f64;
    let mut row_ptrs = vec![0u32];
    let (mut columns, mut values) = (Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            columns.push(j as u32);
            values.push(if i == j {
                (0..n).filter(|&k| k != i).map(|k| weight(i, k)).sum()
            } else {
                -weight(i, j)
            });
        }
        row_ptrs.push(columns.len() as u32);
    }
    (row_ptrs, columns, values)
}

#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn factor_json_roundtrip_preserves_solve(#[case] backend: Backend) {
    let (row_ptrs, columns) = ([0u32, 2, 4, 6, 8], [0u32, 1, 0, 1, 2, 3, 2, 3]);
    let values = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
    let split = CsrRef::new(&row_ptrs, &columns, &values, 4).expect("valid CSR");

    // Strictly dominant, so ingestion grounds it and the restored factor has to
    // recover the augmentation from its blocks' anchors.
    let (sddm_row_ptrs, sddm_columns) = ([0u32, 2, 4], [0u32, 1, 0, 1]);
    let sddm_values = [2.0, -1.0, -1.0, 2.0];
    let sddm = CsrRef::new(&sddm_row_ptrs, &sddm_columns, &sddm_values, 2).expect("valid CSR");

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
        &factorize_with(split, config).expect("factorization should succeed"),
        &[1.0, -1.0, 2.0, -2.0],
    );
    let grounded = factorize_with(sddm, config).expect("factorization should succeed");
    assert_eq!(
        grounded.n(),
        grounded.original_n() + 1,
        "SDDM input augments by one"
    );
    assert_roundtrip("grounded SDDM", &grounded, &[1.0, -1.0]);
}

/// Postcard is how `within` persists a factor, and it is positional — so it, not JSON, is
/// what pins the payload's field order. It also parses an `f64` back exactly, where
/// `serde_json` rounds a 17-digit one by an ulp, so a column long enough for the derived
/// remainder to depend on every share before it can only be checked here.
#[test]
fn a_postcard_roundtrip_reproduces_long_columns_bit_for_bit() {
    let (row_ptrs, columns, values) = complete_laplacian(9);
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 9).expect("valid CSR");
    let factor = factorize_with(
        csr,
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    )
    .expect("factorization should succeed");

    let bytes = postcard::to_stdvec(&factor).expect("serialize factor");
    let restored: Factor<f64> = postcard::from_bytes(&bytes).expect("deserialize factor");

    let b = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 0.0];
    assert_eq!(
        factor.solve(&b).expect("solve original"),
        restored.solve(&b).expect("solve restored"),
        "a restored factor must reproduce the solve bit-for-bit"
    );
}

fn assert_roundtrip(label: &str, factor: &Factor<f64>, b: &[f64]) {
    let json = serde_json::to_string(factor).expect("serialize factor");
    let restored: Factor<f64> = serde_json::from_str(&json).expect("deserialize factor");

    assert_eq!(restored.n(), factor.n(), "{label}");
    assert_eq!(restored.original_n(), factor.original_n(), "{label}");
    assert_eq!(restored.n_steps(), factor.n_steps(), "{label}");
    assert_eq!(
        factor.solve(b).expect("solve original"),
        restored.solve(b).expect("solve restored"),
        "{label}: deserialized factor must reproduce the solve bit-for-bit"
    );
}

#[test]
fn deserializing_corrupted_factor_is_rejected() {
    let mut value = serde_json::to_value(path_factor()).expect("serialize factor");
    assert!(
        value["blocks"][0]["dim"].is_u64(),
        "no block dimension to corrupt"
    );
    value["blocks"][0]["dim"] = serde_json::Value::from(999u32);

    assert!(serde_json::from_value::<Factor<f64>>(value).is_err());
}

/// A payload cannot say that a column hands out more pivot than it has — there is no field
/// for the remainder's share — so the reachable corruption is shares that overspend the
/// pivot between them, which leaves the derived remainder negative.
#[test]
fn a_column_whose_shares_overspend_the_pivot_is_rejected() {
    let (row_ptrs, columns, values) = complete_laplacian(4);
    let csr = CsrRef::new(&row_ptrs, &columns, &values, 4).expect("valid CSR");
    let factor = factorize_with(
        csr,
        Config {
            backend: Backend::Approximate,
            ..Config::default()
        },
    )
    .expect("factorization should succeed");

    let mut value = serde_json::to_value(&factor).expect("serialize factor");
    let shares = &mut value["blocks"][0]["cholesky"]["Approximate"]["steps"][0]["column"]["shares"];
    assert!(
        shares[0][1].is_f64(),
        "the leading step of a complete graph carries shares to overspend"
    );
    shares[0][1] = serde_json::Value::from(2.0);

    assert!(serde_json::from_value::<Factor<f64>>(value).is_err());
}

/// Deliberate, and the cost of deriving the dimension: no wire fact contradicts an anchor
/// any more, so tampering with one answers a different system instead of being an error.
#[test]
fn a_tampered_block_anchor_deserializes_and_answers_a_different_system() {
    let factor = path_factor();
    let mut value = serde_json::to_value(&factor).expect("serialize factor");
    value["blocks"][0]["anchor"] = serde_json::Value::from("Ground");

    let restored: Factor<f64> =
        serde_json::from_value(value).expect("nothing on the wire falsifies an anchor");
    assert_eq!(restored.n(), factor.n());
    assert_eq!(restored.original_n(), factor.original_n() - 1);

    // The anchor decides whether the block's last entry is pinned or projected out, so
    // the tampered factor is not merely one variable short.
    let b = [1.0, 2.0, -3.0];
    let honest = factor.solve(&b).expect("solve the honest factor");
    let tampered = restored.solve(&b).expect("solve the tampered factor");
    assert_ne!(honest[..tampered.len()], tampered[..]);
}

#[test]
fn a_payload_declares_the_format_version_it_was_written_with() {
    let value = serde_json::to_value(path_factor()).expect("serialize factor");
    assert_eq!(
        value["format_version"].as_u64(),
        Some(u64::from(FACTOR_FORMAT_VERSION)),
        "a persisted factor must say which encoding produced it"
    );
}

/// A missing field stands for a payload written before the version existed, so both it
/// and a future encoding have to fail for the version rather than for some interior
/// field a reader cannot act on.
#[rstest]
#[case::from_a_future_release(Some(FACTOR_FORMAT_VERSION + 1))]
#[case::from_before_the_field_existed(None)]
fn a_payload_of_another_format_version_is_rejected_by_version(#[case] declared: Option<u32>) {
    let mut value = serde_json::to_value(path_factor()).expect("serialize factor");
    match declared {
        Some(version) => value["format_version"] = serde_json::Value::from(version),
        None => {
            value
                .as_object_mut()
                .expect("factor serializes as a map")
                .remove("format_version");
        }
    }

    let error = serde_json::from_value::<Factor<f64>>(value)
        .expect_err("a foreign format version must not deserialize")
        .to_string();
    let found = declared.unwrap_or(0);
    assert!(
        error.contains(&format!("format version {found:#010x}")),
        "error must name the version it found, got: {error}"
    );
    assert!(
        error.contains(&format!("{FACTOR_FORMAT_VERSION:#010x}")),
        "error must name the version this build reads, got: {error}"
    );
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

    let json = serde_json::to_string(&config).expect("serialize config");
    let restored: Config = serde_json::from_str(&json).expect("deserialize config");

    assert_eq!(restored.seed, config.seed);
    assert_eq!(restored.split_merge, config.split_merge);
    assert_eq!(restored.backend, config.backend);
}

#[test]
fn config_without_a_backend_deserializes_to_the_default() {
    let restored: Config =
        serde_json::from_str(r#"{"seed":1,"split_merge":null}"#).expect("deserialize config");
    assert_eq!(restored.backend, Backend::default());
}
