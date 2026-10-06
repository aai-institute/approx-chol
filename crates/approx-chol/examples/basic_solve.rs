//! Factorize a 10×10 grid Laplacian and solve a linear system.

#[path = "shared/mod.rs"]
mod shared;

use approx_chol::factorize;
use shared::grid_laplacian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lap = grid_laplacian(10, 10);
    let n = lap.n as usize;
    println!("Grid Laplacian: {}×{} ({} nodes)", 10, 10, n);

    let factor = factorize(lap.as_csr()?)?;
    println!(
        "Factorization: {} elimination steps (factor dimension {})",
        factor.n_steps(),
        factor.n()
    );

    // A Laplacian is singular, so only a zero-sum b has a solution.
    let mut b = vec![0.0f64; n];
    for (i, bi) in b.iter_mut().enumerate() {
        *bi = if i < n / 2 { 1.0 } else { -1.0 };
    }
    assert_eq!(b.iter().sum::<f64>(), 0.0, "RHS must sum to zero");

    let x = factor.solve(&b)?;

    let all_finite = x.iter().all(|v| v.is_finite());
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mean: f64 = x[..n].iter().sum::<f64>() / n as f64;

    println!("Solution: all finite = {all_finite}");
    println!("Solution: L2 norm    = {norm:.6}");
    println!("Solution: mean       = {mean:.2e}  (near zero after gauge fix)");

    Ok(())
}
