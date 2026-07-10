//! Balanceable signed SDD input (#15): folded internally, solved exactly in
//! caller coordinates. A class assertion that the matrix is sign-free rejects a
//! surviving positive off-diagonal instead of silently dropping it (0.2.x).

#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{Config, CsrRef, Error, InputClass};

/// `sddm_solve_is_exact_at_caller_dimension`'s proven-exact SDDM with node 1's
/// sign flipped (a `[1, -1, 1]` congruence): `[[1.5, 1, 0], [1, 2.5, 1], [0, 1,
/// 1.5]]`. Folding recovers the same numerics, so the caller-coordinate solve
/// is exact: `M x = [1, 1, 1]` ⇒ `x = [6/7, -2/7, 6/7]`.
#[test]
fn balanceable_signed_sdd_solves_exactly_at_caller_dimension() {
    let csr = CsrRef::new(
        &[0u32, 2, 5, 7],
        &[0u32, 1, 0, 1, 2, 1, 2],
        &[1.5, 1.0, 1.0, 2.5, 1.0, 1.0, 1.5],
        3,
    )
    .or_panic("valid signed SDD");
    let factor = Builder::new(Config::default())
        .build(csr)
        .or_panic("balanceable signed input should factorize");

    let x = factor.solve(&[1.0, 1.0, 1.0]).or_panic("solve");
    let expected = [6.0f64 / 7.0, -2.0 / 7.0, 6.0 / 7.0];
    assert_eq!(x.len(), 3);
    for (i, (&a, &b)) in x.iter().zip(&expected).enumerate() {
        assert!((a - b).abs() < 1e-12, "x[{i}] = {a}, expected {b}");
    }
}

/// Asserting `Laplacian` skips folding, so a positive off-diagonal is a broken
/// assertion and must error — not silently drop as 0.2.x did.
#[test]
fn asserted_laplacian_rejects_positive_off_diagonal() {
    let csr = CsrRef::new(&[0u32, 2, 4], &[0u32, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2)
        .or_panic("valid csr");
    let err = Builder::new(Config {
        assume: InputClass::Laplacian,
        ..Default::default()
    })
    .build(csr)
    .err_or_panic("positive off-diagonal under assume: Laplacian must error");
    assert!(matches!(err, Error::PositiveOffDiagonal { edge: (0, 1) }));
}

/// Sign-free input under the default `Auto` is the identity pass: no congruence
/// is attached, which is what keeps the widened default byte-compatible.
#[test]
fn sign_free_input_attaches_no_congruence() {
    let csr = CsrRef::new(
        &[0u32, 2, 5, 7],
        &[0u32, 1, 0, 1, 2, 1, 2],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        3,
    )
    .or_panic("valid Laplacian");
    let factor = Builder::<f64>::new(Config::default())
        .build(csr)
        .or_panic("factorize");
    assert!(
        factor.congruence().is_none(),
        "sign-free input must not fold"
    );
}
