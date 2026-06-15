#![cfg(feature = "sprs")]

#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/path.rs"]
mod path;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;

use approx_chol::low_level::Builder;
use approx_chol::{factorize, Config, CsrError, CsrRef, Error};
use num_traits::{Float, FromPrimitive};

/// Build a 4-node path graph Laplacian (0-1-2-3) as a sprs CSR matrix.
fn path_laplacian_sprs<T, I>() -> sprs::CsMatI<T, I>
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: sprs::SpIndex,
{
    let n = path::N as usize;
    let indptr = path::ROW_PTRS.into_iter().map(I::from_usize).collect();
    let indices = path::COL_INDICES.into_iter().map(I::from_usize).collect();
    let data = path::VALUES
        .into_iter()
        .map(|v| T::from_f64(v).or_panic("value conversion"))
        .collect();
    sprs::CsMatI::new((n, n), indptr, indices, data)
}

fn run_case<T, I>()
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: sprs::SpIndex + num_traits::PrimInt + 'static,
{
    let mat = path_laplacian_sprs::<T, I>();
    let csr = CsrRef::try_from(&mat).or_panic("try_from should succeed for valid CSR");

    assert_eq!(csr.n(), 4);
    assert_eq!(csr.row_ptrs().len(), 5);
    assert_eq!(csr.col_indices().len(), 10);
    assert_eq!(csr.values().len(), 10);

    let builder = Builder::<T>::new(Config::default());
    let factor = builder.build(&mat).or_panic("factorization should succeed");

    assert!(factor.n() >= 4);
    assert!(factor.n_steps() > 0);

    let b = [
        T::from_f64(1.0).or_panic("conv"),
        T::from_f64(-1.0).or_panic("conv"),
        T::from_f64(1.0).or_panic("conv"),
        T::from_f64(-1.0).or_panic("conv"),
    ];
    let mut work = vec![T::zero(); factor.n()];
    factor
        .solve_into(&b, &mut work)
        .or_panic("solve_into should succeed");

    assert!(work.iter().all(|x| x.is_finite()));
    let min_signal = T::from_f64(1e-6).or_panic("conv");
    assert!(work.iter().any(|x| x.abs() > min_signal));
}

#[test]
fn sprs_factorize_high_level() {
    let mat = path_laplacian_sprs::<f64, u32>();
    let factor = factorize(&mat).or_panic("factorization should succeed");
    assert!(factor.n() >= 4);
}

#[test]
fn sprs_try_from_is_fallible_and_works() {
    let mat = path_laplacian_sprs::<f64, u64>();
    let csr = CsrRef::try_from(&mat).or_panic("fallible conversion should succeed");
    let factor = factorize(csr).or_panic("factorization should succeed");
    assert!(factor.n() >= 4);
}

#[test]
fn sprs_u32_f64() {
    run_case::<f64, u32>();
}

#[test]
fn sprs_u32_f32() {
    run_case::<f32, u32>();
}

#[test]
fn sprs_usize_f64() {
    run_case::<f64, usize>();
}

#[test]
fn sprs_usize_f32() {
    run_case::<f32, usize>();
}

#[test]
fn sprs_u64_f64() {
    run_case::<f64, u64>();
}

#[test]
fn sprs_u64_f32() {
    run_case::<f32, u64>();
}

#[test]
fn sprs_try_from_csc_returns_error() {
    let csr = path_laplacian_sprs::<f64, u32>();
    let csc = csr.to_csc();
    let err = CsrRef::try_from(csc.view()).err_or_panic("CSC must be rejected");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::ExpectedCsrMatrixGotCsc)
    ));
}

#[test]
fn sprs_factorize_rejects_csc_with_error() {
    let csr = path_laplacian_sprs::<f64, u32>();
    let csc = csr.to_csc();
    let err = factorize(&csc).err_or_panic("CSC must be rejected");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::ExpectedCsrMatrixGotCsc)
    ));
}

#[test]
fn sprs_try_from_non_square_returns_error() {
    let mat = sprs::CsMatI::<f64, u32>::new((3, 4), vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0; 3]);
    let err = CsrRef::try_from(&mat).err_or_panic("non-square matrix must be rejected");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 })
    ));
}
