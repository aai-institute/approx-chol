#[path = "common/backends.rs"]
mod backends;
#[path = "common/grid.rs"]
mod grid;

use approx_chol::{factorize_with, Backend, Config, CsrRef};
use backends::backends;
use num_traits::PrimInt;

fn solution_bits<I: PrimInt + 'static>(input: CsrRef<'_>, backend: Backend) -> Vec<u64> {
    let widen = |indices: &[u32]| -> Vec<I> {
        indices
            .iter()
            .map(|&index| I::from(index).expect("fits"))
            .collect()
    };
    let (row_ptrs, col_indices) = (widen(input.row_ptrs()), widen(input.col_indices()));
    let n = input.n();
    let csr = CsrRef::new(&row_ptrs, &col_indices, input.values(), n as u32).expect("valid CSR");
    let config = Config {
        backend,
        ..Config::default()
    };
    let factor = factorize_with(csr, config).expect("factorizes");
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let x = factor.solve(&b).expect("solves");
    x.into_iter().map(f64::to_bits).collect()
}

/// A grounded component beside a floating one, rewritten or not, and a grid large enough to sample.
#[test]
fn u32_and_u64_input_factor_identically() {
    let grid = grid::grid_laplacian(6, 6);
    let matrices = [
        CsrRef::new(
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0],
            4,
        )
        .expect("valid CSR"),
        // The same matrix with row 0 reordered and its edge split in two: the rewrite path.
        CsrRef::new(
            &[0, 3, 5, 7, 9],
            &[1, 0, 1, 0, 1, 2, 3, 2, 3],
            &[-0.5, 5.0, -0.5, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0],
            4,
        )
        .expect("valid CSR"),
        grid.as_csr().expect("valid CSR"),
    ];
    for backend in backends() {
        for csr in matrices {
            assert_eq!(
                solution_bits::<u64>(csr, backend),
                solution_bits::<u32>(csr, backend),
                "{backend:?} on n = {}",
                csr.n()
            );
        }
    }
}
