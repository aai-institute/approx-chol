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
use num_traits::{Float, FromPrimitive, PrimInt};

fn idx<I: TryFrom<usize>>(value: usize) -> I
where
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
{
    I::try_from(value).or_panic("index conversion")
}

fn path_laplacian<I, T>() -> (Vec<I>, Vec<I>, Vec<T>, u32)
where
    I: PrimInt + TryFrom<usize>,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let row_ptrs = path::ROW_PTRS.into_iter().map(idx::<I>).collect();
    let col_indices = path::COL_INDICES.into_iter().map(idx::<I>).collect();
    let values = path::VALUES
        .into_iter()
        .map(|v| T::from_f64(v).or_panic("value conversion"))
        .collect();
    (row_ptrs, col_indices, values, path::N)
}

/// Factorize the shared path-Laplacian fixture at `config` and check the solve.
fn run_case<I, T>(config: Config)
where
    I: PrimInt + TryFrom<usize> + 'static,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let (rp, ci, vals, n) = path_laplacian::<I, T>();
    let csr = CsrRef::new(&rp, &ci, &vals, n).or_panic("valid csr");
    let factor = Builder::<T>::new(config)
        .build(csr)
        .or_panic("factorization should succeed");

    assert_eq!(factor.n_steps(), factor.n().saturating_sub(1));
    assert_solves_path_rhs(&factor);
}

/// One factorization per (index, scalar) pair, on both the AC and AC2 paths.
#[test]
fn low_level_builder_is_generic_over_index_and_scalar_types() {
    for config in [
        Config::default(),
        Config {
            split_merge: 2,
            seed: 7,
        },
    ] {
        run_case::<u32, f64>(config);
        run_case::<u32, f32>(config);
        run_case::<u64, f64>(config);
        run_case::<u64, f32>(config);
        run_case::<usize, f64>(config);
        run_case::<usize, f32>(config);
    }
}

struct PanicIntoCsr;

impl<'a> From<PanicIntoCsr> for CsrRef<'a, f64, u32> {
    fn from(_: PanicIntoCsr) -> Self {
        panic!("boom during conversion");
    }
}

#[test]
fn factorize_catches_panicking_conversion() {
    let err = factorize::<f64, u32, _>(PanicIntoCsr)
        .err_or_panic("panicking conversion must map to error");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::InputConversionPanicked)
    ));
}
