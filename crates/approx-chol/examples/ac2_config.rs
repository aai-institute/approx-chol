//! AC2 configuration example: compare default AC with the AC2 multi-edge variant.
//!
//! AC2 replaces each edge with `k` copies before factorization and keeps
//! at most `k` copies per neighbor pair after compression. This reduces
//! variance in the approximate factor at the cost of more fill-in per step.
//!
//! Run with:
//! ```
//! cargo run -p approx-chol --example ac2_config
//! ```

#[path = "shared/mod.rs"]
mod shared;

use approx_chol::{factorize_with, Config};
use shared::grid_laplacian;

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lap = grid_laplacian(10, 10);
    let n = lap.n as usize;
    println!("Grid Laplacian: 10×10 ({n} nodes)\n");

    // -----------------------------------------------------------------------
    // Default AC (split_merge = None)
    // -----------------------------------------------------------------------
    let ac_config = Config::default();
    let ac_factor = factorize_with(lap.sddm(), ac_config)?;

    println!("=== Default AC ===");
    println!("  split_merge : None (standard AC)");
    println!("  n_steps     : {}", ac_factor.n_steps());

    // -----------------------------------------------------------------------
    // AC2 (k = 2)
    // -----------------------------------------------------------------------
    let ac2_config = Config {
        split_merge: Some(2),
        seed: 42,
        ..Config::default()
    };
    let ac2_factor = factorize_with(lap.sddm(), ac2_config)?;

    println!("\n=== AC2 (k=2) ===");
    println!("  split_merge : Some(2)");
    println!("  n_steps     : {}", ac2_factor.n_steps());

    // -----------------------------------------------------------------------
    // Solve the same system with both factors and compare
    // -----------------------------------------------------------------------
    let mut b = vec![0.0f64; n];
    for (i, bi) in b.iter_mut().enumerate() {
        *bi = if i < n / 2 { 1.0 } else { -1.0 };
    }

    let x_ac = ac_factor.solve(&b)?;
    let x_ac2 = ac2_factor.solve(&b)?;

    let norm_ac: f64 = x_ac.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_ac2: f64 = x_ac2.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("\nSolution norms:");
    println!("  AC  : {norm_ac:.6}");
    println!("  AC2 : {norm_ac2:.6}");
    println!("  (norms differ — approximate factors have different spectral quality)");

    Ok(())
}
