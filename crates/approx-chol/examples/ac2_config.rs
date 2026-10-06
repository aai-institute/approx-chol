//! AC vs AC2, whose `k` copies per edge trade more fill-in for lower factor variance.

#[path = "shared/mod.rs"]
mod shared;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use shared::grid_laplacian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lap = grid_laplacian(10, 10);
    let n = lap.n as usize;
    println!("Grid Laplacian: 10×10 ({n} nodes)\n");

    let ac_config = Config::default();
    let ac_factor = Builder::new(ac_config).build(lap.as_csr()?)?;

    println!("=== Default AC ===");
    println!("  split_merge : None (standard AC)");
    println!("  n_steps     : {}", ac_factor.n_steps());
    println!("  factor dim  : {}", ac_factor.n());

    let ac2_config = Config {
        split_merge: Some(2),
        seed: 42,
        ..Config::default()
    };
    let ac2_factor = Builder::new(ac2_config).build(lap.as_csr()?)?;

    println!("\n=== AC2 (k=2) ===");
    println!("  split_merge : Some(2)");
    println!("  n_steps     : {}", ac2_factor.n_steps());
    println!("  factor dim  : {}", ac2_factor.n());

    // AC2 changes only how edges are sampled, not the factor dimension.
    assert_eq!(
        ac_factor.n(),
        ac2_factor.n(),
        "AC and AC2 must have the same factor dimension"
    );
    println!(
        "\nNote: factor dimensions match ({}) — AC2 changes sampling quality, not matrix size.",
        ac_factor.n()
    );

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
