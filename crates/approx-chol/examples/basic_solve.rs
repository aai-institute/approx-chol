//! Factorize a 10×10 grid Laplacian with AC and AC2, and use each to precondition CG.

#[path = "shared/grid.rs"]
mod grid;

use approx_chol::input::Laplacian;
use approx_chol::{factorize_with, Config};
use grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(10, 10);
    println!("Grid Laplacian: 10×10 ({} vertices)", grid.n);

    let laplacian = Laplacian::<f64>::new(grid.n, &grid.a, &grid.b)?;
    let b = grid.split_rhs();

    // AC2's `k` copies per edge trade more fill-in for lower factor variance.
    let ac2 = Config {
        split_merge: Some(2),
        ..Config::default()
    };
    for (name, config) in [("AC", Config::default()), ("AC2 (k=2)", ac2)] {
        let factor = factorize_with(laplacian, config)?;
        let (x, iterations) = grid.pcg(&factor, &b, 1e-10);
        let mean = x.iter().sum::<f64>() / grid.n as f64;
        let blocks = factor.blocks().len();
        println!(
            "{name:>9}: {blocks} block(s), {iterations} PCG iterations, mean {mean:.2e} (⟂ 1)"
        );
    }

    Ok(())
}
