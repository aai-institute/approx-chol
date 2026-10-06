//! Its own binary, since dhat's allocator would slow every other suite.

#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{factorize, CsrRef, Error};
use laplacian_prop::{build_laplacian_csr, LaplacianCsr};
use num_traits::PrimInt;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const N: usize = 32;

/// The last row's deficit makes ingestion run to its verdict and stop before any factorization.
fn deficient(edge_weights: &[u8]) -> LaplacianCsr {
    let (row_ptrs, col_indices, mut values, n) = build_laplacian_csr(N, edge_weights);
    // Columns ascend, so the last row's diagonal is its last entry.
    values[row_ptrs[N] as usize - 1] -= 0.5;
    (row_ptrs, col_indices, values, n)
}

fn path() -> LaplacianCsr {
    let mut weights = vec![0u8; N * (N - 1) / 2];
    let mut pair = 0;
    for i in 0..N {
        for j in i + 1..N {
            if j == i + 1 {
                weights[pair] = 1;
            }
            pair += 1;
        }
    }
    deficient(&weights)
}

fn complete() -> LaplacianCsr {
    deficient(&vec![1u8; N * (N - 1) / 2])
}

/// Heap bytes and blocks ingestion allocates for `csr`, widened to `I`.
fn ingestion_allocations<I: PrimInt + 'static>(csr: &LaplacianCsr) -> (u64, u64) {
    let (row_ptrs, col_indices, values, n) = csr;
    let widen = |indices: &[u32]| -> Vec<I> {
        indices
            .iter()
            .map(|&index| I::from(index).expect("fits"))
            .collect()
    };
    let (row_ptrs, col_indices) = (widen(row_ptrs), widen(col_indices));
    let csr = CsrRef::new(&row_ptrs, &col_indices, values, *n).expect("valid CSR");

    let before = dhat::HeapStats::get();
    let result = factorize(csr);
    let after = dhat::HeapStats::get();
    assert!(matches!(
        result,
        Err(Error::NotDiagonallyDominant { row }) if row == N - 1
    ));
    (
        after.total_bytes - before.total_bytes,
        after.total_blocks - before.total_blocks,
    )
}

#[test]
fn ingesting_canonical_input_allocates_nothing_per_nonzero() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let (path, complete) = (path(), complete());
    let baseline = ingestion_allocations::<u32>(&path);

    assert_eq!(
        ingestion_allocations::<u32>(&complete),
        baseline,
        "a denser matrix of the same dimension allocated more"
    );
    assert_eq!(ingestion_allocations::<u64>(&path), baseline);
    assert_eq!(ingestion_allocations::<u64>(&complete), baseline);
}
