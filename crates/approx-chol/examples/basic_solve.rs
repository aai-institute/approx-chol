//! Basic example: factorize a 10×10 grid Laplacian and solve a linear system.
//!
//! Run with:
//! ```
//! cargo run -p approx-chol --example basic_solve
//! ```

#[path = "shared/mod.rs"]
mod shared;

use approx_chol::factorize;
use shared::grid_laplacian;

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
