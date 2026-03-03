//! Basic example: factorize a 10×10 grid Laplacian and solve a linear system.
//!
//! Run with:
//! ```
//! cargo run -p approx-chol --example basic_solve
//! ```

use approx_chol::{factorize, CsrRef, Error};

// --------------------------------------------------------------------------
// Grid Laplacian builder (inlined — examples are separate compilation units)
// --------------------------------------------------------------------------

struct GridLaplacian {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
    n: u32,
}

impl GridLaplacian {
    fn as_csr(&self) -> Result<CsrRef<'_>, Error> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n)
    }
}

/// Build an `rows × cols` 2-D grid Laplacian.
///
/// Interior nodes have degree 4, boundary nodes degree 3, corners degree 2.
/// The matrix is stored in CSR format with sorted column indices per row.
fn grid_laplacian(rows: usize, cols: usize) -> GridLaplacian {
    let n = rows * cols;
    let mut row_ptrs: Vec<u32> = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);

    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            let mut diag = 0.0f64;
            let mut neighbors: Vec<usize> = Vec::new();

            if r > 0 {
                neighbors.push((r - 1) * cols + c);
                diag += 1.0;
            }
            if c > 0 {
                neighbors.push(r * cols + c - 1);
                diag += 1.0;
            }
            let diag_pos = neighbors.len();
            neighbors.push(v);
            if c + 1 < cols {
                neighbors.push(r * cols + c + 1);
                diag += 1.0;
            }
            if r + 1 < rows {
                neighbors.push((r + 1) * cols + c);
                diag += 1.0;
            }

            for (i, &nbr) in neighbors.iter().enumerate() {
                col_indices.push(nbr as u32);
                values.push(if i == diag_pos { diag } else { -1.0 });
            }
            row_ptrs.push(col_indices.len() as u32);
        }
    }

    GridLaplacian {
        row_ptrs,
        col_indices,
        values,
        n: n as u32,
    }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a 10×10 grid Laplacian (100 nodes).
    let lap = grid_laplacian(10, 10);
    let n = lap.n as usize;
    println!("Grid Laplacian: {}×{} ({} nodes)", 10, 10, n);

    // Factorize with default configuration (AC, DynamicPQ ordering).
    let factor = factorize(lap.as_csr()?)?;
    println!(
        "Factorization: {} elimination steps (factor dimension {})",
        factor.n_steps(),
        factor.n()
    );

    // Build a zero-sum right-hand side: b[i] = +1 for i < n/2, -1 for i >= n/2.
    // Laplacians are singular; a solution exists only when b is in the column space
    // (i.e., sum(b) == 0).
    let mut b = vec![0.0f64; n];
    for (i, bi) in b.iter_mut().enumerate() {
        *bi = if i < n / 2 { 1.0 } else { -1.0 };
    }
    assert_eq!(b.iter().sum::<f64>(), 0.0, "RHS must sum to zero");

    // Solve: returns a newly allocated vector of length factor.n().
    let x = factor.solve(&b)?;

    // Quality check.
    let all_finite = x.iter().all(|v| v.is_finite());
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mean: f64 = x[..n].iter().sum::<f64>() / n as f64;

    println!("Solution: all finite = {all_finite}");
    println!("Solution: L2 norm    = {norm:.6}");
    println!("Solution: mean       = {mean:.2e}  (near zero after gauge fix)");

    Ok(())
}
