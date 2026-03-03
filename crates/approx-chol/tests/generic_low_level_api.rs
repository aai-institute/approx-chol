use approx_chol::{factorize, Builder, Config, CsrRef, Error};
use num_traits::{Float, FromPrimitive, PrimInt};

fn idx<I: TryFrom<usize>>(value: usize) -> I
where
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
{
    I::try_from(value).expect("index conversion")
}

fn path_laplacian<I, T>() -> (Vec<I>, Vec<I>, Vec<T>, u32)
where
    I: PrimInt + TryFrom<usize>,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let row_ptrs = [0usize, 2, 5, 8, 10].into_iter().map(idx::<I>).collect();
    let col_indices = [0usize, 1, 0, 1, 2, 1, 2, 3, 2, 3]
        .into_iter()
        .map(idx::<I>)
        .collect();
    let values = [1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0]
        .into_iter()
        .map(|v| T::from_f64(v).expect("value conversion"))
        .collect();
    (row_ptrs, col_indices, values, 4)
}

fn run_case<I, T>()
where
    I: PrimInt + TryFrom<usize> + 'static,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let (rp, ci, vals, n) = path_laplacian::<I, T>();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid csr");

    let factor = factorize(csr).expect("factorization should succeed");
    assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));

    let b = [
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
    ];
    let mut work = vec![T::zero(); factor.n()];
    factor
        .solve_into(&b, &mut work)
        .expect("solve_into should succeed");

    assert!(work.iter().all(|x| x.is_finite()));
    let min_signal = T::from_f64(1e-6).expect("conv");
    assert!(work.iter().any(|x| x.abs() > min_signal));
}

fn run_case_ac2<I, T>()
where
    I: PrimInt + TryFrom<usize> + 'static,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let (rp, ci, vals, n) = path_laplacian::<I, T>();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid csr");
    let builder = Builder::<T>::new(Config {
        split_merge: Some(2),
        seed: 7,
    });
    let factor = builder
        .build(csr)
        .expect("AC2 generic factorization should succeed");

    let b = [
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
        T::from_f64(1.0).expect("conv"),
        T::from_f64(-1.0).expect("conv"),
    ];
    let mut work = vec![T::zero(); factor.n()];
    factor
        .solve_into(&b, &mut work)
        .expect("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
}

#[test]
fn low_level_u32_f64() {
    run_case::<u32, f64>();
}

#[test]
fn low_level_u32_f32() {
    run_case::<u32, f32>();
}

#[test]
fn low_level_u64_f64() {
    run_case::<u64, f64>();
}

#[test]
fn low_level_u64_f32() {
    run_case::<u64, f32>();
}

#[test]
fn low_level_usize_f64() {
    run_case::<usize, f64>();
}

#[test]
fn low_level_usize_f32() {
    run_case::<usize, f32>();
}

#[test]
fn low_level_default_factorize_u32_no_conversion_path() {
    let rp = [0u32, 2, 5, 8, 10];
    let ci = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let vals = [1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&rp, &ci, &vals, 4).expect("valid csr");
    let factor = factorize(csr).expect("factorization should succeed");

    let b = [1.0_f64, -1.0, 1.0, -1.0];
    let mut work = vec![0.0; factor.n()];
    factor
        .solve_into(&b, &mut work)
        .expect("solve_into should succeed");
    assert!(work.iter().all(|x| x.is_finite()));
}

#[test]
fn low_level_factorize_csrref() {
    let rp = [0u32, 2, 5, 8, 10];
    let ci = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let vals = [1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&rp, &ci, &vals, 4).expect("valid csr");
    let factor = factorize(csr).expect("factorization should succeed");
    assert_eq!(factor.n(), 4);
}

#[test]
fn low_level_u64_f32_ac2() {
    run_case_ac2::<u64, f32>();
}

#[test]
fn low_level_usize_f64_ac2() {
    run_case_ac2::<usize, f64>();
}

#[test]
fn split_zero_is_rejected() {
    let rp = [0u32, 2, 5, 8, 10];
    let ci = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let vals = [1.0_f64, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&rp, &ci, &vals, 4).expect("valid csr");
    let builder = Builder::<f64>::new(Config {
        split_merge: Some(0),
        ..Default::default()
    });
    let err = builder
        .build(csr)
        .expect_err("split_merge=0 should return InvalidConfig");
    assert!(matches!(
        err,
        Error::InvalidConfig("split_merge must be >= 1")
    ));
}

struct PanicIntoCsr;

impl<'a> From<PanicIntoCsr> for CsrRef<'a, f64, u32> {
    fn from(_: PanicIntoCsr) -> Self {
        panic!("boom during conversion");
    }
}

#[test]
fn factorize_catches_panicking_conversion() {
    let err =
        factorize::<f64, u32, _>(PanicIntoCsr).expect_err("panicking conversion must map to error");
    assert!(matches!(
        err,
        Error::InvalidCsr("input conversion panicked")
    ));
}
