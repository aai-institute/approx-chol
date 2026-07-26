#![cfg(feature = "faer")]

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
use faer::sparse::SparseRowMat;
use num_traits::{cast, Float, FromPrimitive, PrimInt};

/// Build a 4-node path graph Laplacian (0-1-2-3) as a faer sparse CSR matrix.
fn path_laplacian_faer<T, I>() -> SparseRowMat<I, T>
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: faer::Index + PrimInt,
{
    let nrows = path::N as usize;
    let ncols = path::N as usize;
    let row_ptrs = path::ROW_PTRS
        .into_iter()
        .map(|v| cast::<usize, I>(v).or_panic("index conversion"))
        .collect();
    let col_indices = path::COL_INDICES
        .into_iter()
        .map(|v| cast::<usize, I>(v).or_panic("index conversion"))
        .collect();
    let values = path::VALUES
        .into_iter()
        .map(|v| T::from_f64(v).or_panic("value conversion"))
        .collect();

    let symbolic = faer::sparse::SymbolicSparseRowMat::<I>::new_checked(
        nrows,
        ncols,
        row_ptrs,
        None,
        col_indices,
    );
    SparseRowMat::new(symbolic, values)
}

fn run_case<T, I>()
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: faer::Index + PrimInt + 'static,
{
    let mat = path_laplacian_faer::<T, I>();
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
fn faer_csr_factorizes_over_index_and_scalar_types() {
    run_case::<f64, u32>();
    run_case::<f32, u32>();
    run_case::<f64, usize>();
    run_case::<f32, usize>();
    run_case::<f64, u64>();
    run_case::<f32, u64>();
}

#[test]
fn faer_factorize_rejects_non_square_with_error() {
    let symbolic = faer::sparse::SymbolicSparseRowMat::<u32>::new_checked(
        3,
        4,
        vec![0u32, 1, 2, 3],
        None,
        vec![0u32, 1, 0],
    );
    let mat = SparseRowMat::new(symbolic, vec![1.0, 1.0, 1.0]);
    let err = factorize(&mat).err_or_panic("non-square matrix must be rejected");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::ExpectedSquareMatrix { rows: 3, cols: 4 })
    ));
}
