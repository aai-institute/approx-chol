//! AC vs AC2, whose `k` copies per edge trade more fill-in for lower factor variance.

#[path = "shared/grid.rs"]
mod grid;

use approx_chol::input::{Laplacian, Weights};
use approx_chol::{factorize_with, Config};
use grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(10, 10);
    println!("Grid Laplacian: 10×10 ({} vertices)", grid.n);

    let laplacian = Laplacian::new(grid.n, &grid.a, &grid.b, Weights::<f64>::unit())?;
    let b = grid.split_rhs();

    let configs = [
        ("AC", Config::default()),
        (
            "AC2 (k=2)",
            Config {
                split_merge: Some(2),
                ..Config::default()
            },
        ),
    ];
    for (name, config) in configs {
        let factor = factorize_with(laplacian, config)?;
        let (_, iterations) = grid.pcg(&factor, &b, 1e-10);
        println!("{name:>9}: {iterations} PCG iterations to 1e-10");
    }

    Ok(())
}
