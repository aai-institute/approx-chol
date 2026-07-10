//! Augmentation measures the deficit it clamps (#8).
//!
//! Gremban augmentation floors each negative row sum to zero instead of
//! grounding it — the crate's one silent approximation. The built factor now
//! reports that clamped mass (total and worst row) exactly, and reports exactly
//! zero when the input is dominant so nothing is clamped.

#[path = "common/panic_ok.rs"]
mod panic_ok;
use panic_ok::OrPanic;

use approx_chol::{factorize, CsrRef};

/// Injects a known deficit on two rows of different magnitude (rows 1 and 2,
/// deficits 3 and 1) flanked by surplus rows so augmentation is well-formed.
/// `total` sums the clamped mass (4); `worst_row` is the max (3) — distinct
/// values, so this also proves `worst_row` is a max and not a second sum.
#[test]
fn injected_deficit_is_reported_exactly() {
    let csr = CsrRef::new(
        &[0u32, 2, 5, 8, 10],
        &[0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        &[2.0, -1.0, -1.0, 2.0, -4.0, -4.0, 4.0, -1.0, -1.0, 2.0],
        4,
    )
    .or_panic("valid csr");
    let factor = factorize(csr).or_panic("non-dominant input still factorizes (clamped)");

    let deficit = factor.deficit();
    assert_eq!(deficit.total, 4.0, "total clamped mass");
    assert_eq!(deficit.worst_row, 3.0, "worst single-row clamp");
}

/// A dominant input (path Laplacian, zero row sums) clamps nothing, so the
/// reported deficit is exactly zero — not an epsilon-thresholded near-zero.
#[test]
fn dominant_input_reports_zero_deficit() {
    let csr = CsrRef::new(
        &[0u32, 2, 5, 8, 10],
        &[0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        4,
    )
    .or_panic("valid Laplacian");
    let factor = factorize::<f64, _, _>(csr).or_panic("factorize");

    let deficit = factor.deficit();
    assert_eq!(deficit.total, 0.0);
    assert_eq!(deficit.worst_row, 0.0);
}
