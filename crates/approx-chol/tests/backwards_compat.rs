#![cfg(feature = "serde")]
//! Backwards-compatibility harness for the H-matrix epic (issue #13).
//!
//! Freezes 0.2.x factorization/solve behavior on previously-valid input so
//! every later slice (#14–#20) can prove it stays a no-op there — the epic's
//! core promise is "provably the identity on previously-valid input", and the
//! proof must exist before the changes do.
//!
//! Strict byte-for-byte golden checks run only on macOS (the maintainer's
//! platform): float and RNG formatting are not guaranteed identical across
//! targets, so on other OSes these checks are absent by construction. After an
//! *intended* change, regenerate the goldens with
//! `UPDATE_GOLDEN=1 cargo test -p approx-chol --all-features`.

#[path = "common/panic_ok.rs"]
mod panic_ok;

use approx_chol::{factorize, CsrRef};
use panic_ok::OrPanic;

/// The one intended divergence: input with a positive off-diagonal, which
/// 0.2.x ingestion silently drops — corrupting the factor. Ignored here (no
/// frozen value); #6 routes such entries to a sign policy and #15 makes this a
/// correctness assertion, at which point this test is un-ignored and inverted.
#[test]
#[ignore = "documents the 0.2.x silent-corruption bug; #6/#15 replace this with a correctness assertion"]
fn positive_off_diagonal_is_silently_corrupted() {
    // Symmetric 2x2 with a +1 off-diagonal (not an SDDM edge). Today this
    // succeeds by dropping the positive entry; the factor no longer represents
    // the input matrix.
    let csr = CsrRef::new(&[0u32, 2, 4], &[0u32, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2)
        .or_panic("valid csr");
    let _ = factorize(csr);
}

#[cfg(target_os = "macos")]
#[path = "common/path.rs"]
mod path;

#[cfg(target_os = "macos")]
mod golden {
    use super::panic_ok::OrPanic;
    use super::path;
    use approx_chol::{factorize, CsrRef, Factor};
    use num_traits::{Float, FromPrimitive};

    /// A type-neutral CSR fixture `(row_ptrs, col_indices, values, n)`. Values
    /// are stored once as `f64` and cast to the scalar under test.
    type Fixture = (&'static [usize], &'static [usize], &'static [f64], u32);

    const PATH: Fixture = (&path::ROW_PTRS, &path::COL_INDICES, &path::VALUES, path::N);
    /// Strict-surplus SDDM: 3-node path with +0.5 diagonal surplus per row, so
    /// it is strictly diagonally dominant and exercises Gremban augmentation.
    const SDDM: Fixture = (
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[1.5, -1.0, -1.0, 2.5, -1.0, -1.0, 1.5],
        3,
    );

    fn factor<T: Float + FromPrimitive + Send + Sync + 'static>(
        (row_ptrs, col_indices, values, n): Fixture,
    ) -> Factor<T> {
        let values: Vec<T> = values
            .iter()
            .map(|&v| T::from_f64(v).or_panic("value conversion"))
            .collect();
        factorize(CsrRef::new(row_ptrs, col_indices, &values, n).or_panic("valid csr"))
            .or_panic("factorize")
    }

    fn to_json<T: serde::Serialize>(value: &T) -> String {
        serde_json::to_string_pretty(value).or_panic("serialize to json")
    }

    fn check_golden(name: &str, actual: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden")
            .join(name);
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            std::fs::create_dir_all(path.parent().or_panic("golden path has a parent"))
                .or_panic("create golden dir");
            std::fs::write(&path, actual).or_panic("write golden");
            return;
        }
        let expected = std::fs::read_to_string(&path)
            .or_panic("read golden (create missing goldens with UPDATE_GOLDEN=1)");
        assert_eq!(
            actual, expected,
            "golden {name} drifted; regenerate with UPDATE_GOLDEN=1 if intended"
        );
    }

    /// Declares one `#[test]` per line, each freezing `to_json(&value)` as a
    /// golden named after the test (`<test_name>.json`).
    macro_rules! goldens {
        ($($name:ident => $value:expr;)*) => { $(
            #[test]
            fn $name() {
                check_golden(concat!(stringify!($name), ".json"), &to_json(&$value));
            }
        )* };
    }

    goldens! {
        factor_path_laplacian_f64 => factor::<f64>(PATH);
        factor_path_laplacian_f32 => factor::<f32>(PATH);
        factor_sddm_surplus_f64  => factor::<f64>(SDDM);
        factor_sddm_surplus_f32  => factor::<f32>(SDDM);
        solve_path_laplacian_f64 => factor::<f64>(PATH).solve(&[1.0, -1.0, 1.0, -1.0]).or_panic("solve");
        solve_path_laplacian_f32 => factor::<f32>(PATH).solve(&[1.0, -1.0, 1.0, -1.0]).or_panic("solve");
        // Exact grounded SDDM solve at caller dimension (#14).
        solve_sddm_surplus_f64 => factor::<f64>(SDDM).solve(&[1.0, -1.0, 1.0]).or_panic("solve");
        solve_sddm_surplus_f32 => factor::<f32>(SDDM).solve(&[1.0, -1.0, 1.0]).or_panic("solve");
    }

    /// The SDDM fixture must actually trigger Gremban augmentation, or the
    /// goldens above would silently degenerate into the plain-Laplacian path.
    #[test]
    fn sddm_fixture_triggers_augmentation() {
        let factor = factor::<f64>(SDDM);
        assert_eq!(factor.n(), factor.original_n() + 1);
    }
}
