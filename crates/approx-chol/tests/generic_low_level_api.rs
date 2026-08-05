#[path = "common/path.rs"]
mod path;
#[path = "common/path_solve.rs"]
mod path_solve;
use path_solve::assert_view_and_factor_match_fixture;

use approx_chol::low_level::Builder;
use approx_chol::{factorize, Config, CsrError, CsrRef, Error};
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
    let row_ptrs = path::ROW_PTRS.into_iter().map(idx::<I>).collect();
    let col_indices = path::COL_INDICES.into_iter().map(idx::<I>).collect();
    let values = path::VALUES
        .into_iter()
        .map(|v| T::from_f64(v).expect("value conversion"))
        .collect();
    (row_ptrs, col_indices, values, path::N)
}

fn run_case<I, T>(config: Config)
where
    I: PrimInt + TryFrom<usize> + 'static,
    <I as TryFrom<usize>>::Error: core::fmt::Debug,
    T: Float + FromPrimitive + core::fmt::Debug + Send + Sync + 'static + core::iter::Sum<T>,
{
    let (rp, ci, vals, n) = path_laplacian::<I, T>();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid csr");
    assert_view_and_factor_match_fixture(csr, config);
}

/// One factorization per (index, scalar) pair, on both the AC and AC2 paths.
#[test]
fn low_level_builder_is_generic_over_index_and_scalar_types() {
    for config in [
        Config::default(),
        Config {
            split_merge: Some(2),
            seed: 7,
            ..Config::default()
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
    let err =
        factorize::<f64, u32, _>(PanicIntoCsr).expect_err("panicking conversion must map to error");
    assert!(matches!(
        err,
        Error::InvalidCsr(CsrError::InputConversionPanicked)
    ));
}

/// One code path, so these must agree entry for entry, not merely to roundoff.
#[test]
fn split_below_two_is_standard_ac() {
    let (rp, ci, vals, n) = path_laplacian::<u32, f64>();
    let csr = CsrRef::new(&rp, &ci, &vals, n).expect("valid csr");
    let factor = |split_merge| {
        Builder::<f64>::new(Config {
            split_merge,
            ..Default::default()
        })
        .build(csr)
        .expect("standard AC builds")
    };
    let reference = factor(None);
    let mut b = vec![0.0; n as usize];
    b[0] = 1.0;
    let mut expected = b.clone();
    reference.solve_in_place(&mut expected).expect("solve");
    for split_merge in [Some(0), Some(1)] {
        let mut actual = b.clone();
        factor(split_merge)
            .solve_in_place(&mut actual)
            .expect("solve");
        assert_eq!(actual, expected, "split_merge {split_merge:?} is not AC");
    }
}
