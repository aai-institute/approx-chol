use approx_chol::{factorize_with, Backend, Config, CsrRef};
use num_traits::Float;
use rstest::rstest;

const N: usize = 6;

/// Complete graph on [`N`] vertices at uniform conductance `w`, so scaling `w` scales the
/// whole matrix and nothing else.
fn scaled_solve<T>(w: T, backend: Backend) -> Vec<T>
where
    T: Float + Send + Sync + 'static,
{
    let mut row_ptrs = vec![0u32];
    let (mut col_indices, mut values) = (Vec::new(), Vec::new());
    let degree = T::from(N - 1).expect("small count is representable");
    for row in 0..N {
        for col in 0..N {
            col_indices.push(col as u32);
            values.push(if row == col { w * degree } else { -w });
        }
        row_ptrs.push(col_indices.len() as u32);
    }
    let rhs: Vec<T> = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        .iter()
        .map(|&b| T::from(b).expect("literal is representable"))
        .collect();
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, N as u32).expect("valid csr");
    let config = Config {
        backend,
        ..Config::default()
    };
    factorize_with(csr, config)
        .expect("factorization should succeed")
        .solve(&rhs)
        .expect("solve should succeed")
        .iter()
        .map(|&x| x * w)
        .collect()
}

/// `M x = b` scaled by `w` has solution `x / w`, so `w * x(w)` is invariant for any
/// scale-free method at a fixed seed. An absolute floor in the sampler shows up here as a
/// deviation of order one rather than of order epsilon: before #92, `f64` broke at `1e-14`
/// and `f32` at `1e-6`, both by 98%.
///
/// Above unit scale it is the solve kernel rather than the sampler that the exponents
/// bound: before #93 the pivot entries, at `1/w` of the right-hand side's scale, were
/// annihilated by the residue the uneliminated vertex carried, which `Anchor::recover`
/// then turned into exact zeros.
fn assert_invariant_under_scaling<T>(backend: Backend, exponents: &[i32], tolerance: T)
where
    T: Float + Send + Sync + std::fmt::LowerExp + 'static,
{
    let ten = T::from(10.0).expect("ten is representable");
    let reference = scaled_solve(T::one(), backend);
    for &exponent in exponents {
        let w = ten.powi(exponent);
        for (index, (&scaled, &want)) in scaled_solve(w, backend).iter().zip(&reference).enumerate()
        {
            let deviation = ((scaled - want) / want).abs();
            assert!(
                deviation < tolerance,
                "w=1e{exponent}: x[{index}] deviates by {deviation:e}"
            );
        }
    }
}

#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::exact(Backend::default())]
fn factorization_is_invariant_under_uniform_scaling(#[case] backend: Backend) {
    // `f64`'s floor was `1e-14`, and its annihilation set in at `1e21` on this fixture;
    // the exponents above unit scale are ones that failed before #93.
    assert_invariant_under_scaling(
        backend,
        &[
            -300, -200, -100, -30, -16, -15, -14, -5, -1, 21, 23, 31, 32, 37, 43, 50, 100, 152,
            200, 300,
        ],
        1e-12f64,
    );
    // `f32`'s floor was `1e-6` and its annihilation set in at `1e10`, both ordinary
    // conductance territory.
    assert_invariant_under_scaling(
        backend,
        &[-30, -20, -12, -8, -7, -6, -5, -2, -1, 7, 10, 12, 20, 30],
        1e-4f32,
    );
}
