#![cfg(feature = "sprs")]

#[path = "common/panic_err.rs"]
mod panic_err;
#[path = "common/panic_ok.rs"]
mod panic_ok;
#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use panic_err::ErrOrPanic;
use panic_ok::OrPanic;
use path_solve::assert_solves_path_rhs;

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
    assert_solves_path_rhs(&factor);
}

/// One factorization per (index, scalar) pair the adapter supports.
#[test]
fn sprs_csr_factorizes_over_index_and_scalar_types() {
    run_case::<f64, u32>();
    run_case::<f32, u32>();
    run_case::<f64, usize>();
    run_case::<f32, usize>();
    run_case::<f64, u64>();
    run_case::<f32, u64>();
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
