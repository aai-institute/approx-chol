#![cfg(feature = "faer")]

use approx_chol::{factorize, Builder, Config, CsrRef, Error};
use faer::sparse::SparseRowMat;
use num_traits::{cast, Float, FromPrimitive, PrimInt};

/// Build a 4-node path graph Laplacian (0-1-2-3) as a faer sparse CSR matrix.
fn path_laplacian_faer<T, I>() -> SparseRowMat<I, T>
where
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
    I: faer::Index + PrimInt,
{
    let nrows = 4usize;
    let ncols = 4usize;
    let row_ptrs = [0usize, 2, 5, 8, 10]
        .into_iter()
        .map(|v| cast::<usize, I>(v).expect("index conversion"))
        .collect();
    let col_indices = [0usize, 1, 0, 1, 2, 1, 2, 3, 2, 3]
        .into_iter()
        .map(|v| cast::<usize, I>(v).expect("index conversion"))
        .collect();
    let values = [1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0]
        .into_iter()
        .map(|v| T::from_f64(v).expect("value conversion"))
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
    let csr: CsrRef<'_, T, I> = (&mat).into();

    assert_eq!(csr.n(), 4);
    assert_eq!(csr.row_ptrs().len(), 5);
    assert_eq!(csr.col_indices().len(), 10);
    assert_eq!(csr.values().len(), 10);

    let builder = Builder::<T>::new(Config::default());
    let factor = builder.build(&mat).expect("factorization should succeed");

    assert!(factor.n() >= 4);
    assert!(factor.n_steps() > 0);

    let b = [
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
    ];
    let mut work = vec![T::zero(); factor.n()];
    factor.solve_into(&b, &mut work);

    assert!(work.iter().all(|x| x.is_finite()));
    let min_signal = T::from_f64(1e-6).expect("conv");
    assert!(work.iter().any(|x| x.abs() > min_signal));
}

#[test]
fn faer_factorize_high_level() {
    let mat = path_laplacian_faer::<f64, u32>();
    let factor = factorize(&mat).expect("factorization should succeed");
    assert!(factor.n() >= 4);
}

#[test]
fn faer_try_from_is_fallible_and_works() {
    let mat = path_laplacian_faer::<f64, usize>();
    let csr = CsrRef::try_from_faer(&mat).expect("fallible conversion should succeed");
    let factor = factorize(csr).expect("factorization should succeed");
    assert!(factor.n() >= 4);
}

#[test]
fn faer_u32_f64() {
    run_case::<f64, u32>();
}

#[test]
fn faer_u32_f32() {
    run_case::<f32, u32>();
}

#[test]
fn faer_usize_f64() {
    run_case::<f64, usize>();
}

#[test]
fn faer_usize_f32() {
    run_case::<f32, usize>();
}

#[test]
fn faer_u64_f64() {
    run_case::<f64, u64>();
}

#[test]
fn faer_u64_f32() {
    run_case::<f32, u64>();
}

#[test]
#[should_panic(expected = "expected square matrix")]
fn faer_non_square_panics() {
    let row_ptrs = vec![0u32, 1, 2, 3];
    let col_indices = vec![0u32, 1, 0];
    let values = vec![1.0, 1.0, 1.0];
    let symbolic =
        faer::sparse::SymbolicSparseRowMat::<u32>::new_checked(3, 4, row_ptrs, None, col_indices);
    let mat = SparseRowMat::new(symbolic, values);
    let _: CsrRef<'_, f64, u32> = mat.as_ref().into();
}

#[test]
fn faer_try_from_non_square_returns_error() {
    let row_ptrs = vec![0u32, 1, 2, 3];
    let col_indices = vec![0u32, 1, 0];
    let values = vec![1.0, 1.0, 1.0];
    let symbolic =
        faer::sparse::SymbolicSparseRowMat::<u32>::new_checked(3, 4, row_ptrs, None, col_indices);
    let mat = SparseRowMat::new(symbolic, values);
    let err = CsrRef::try_from_faer(&mat).expect_err("non-square matrix must be rejected");
    assert!(matches!(err, Error::InvalidCsr("expected square matrix")));
}
