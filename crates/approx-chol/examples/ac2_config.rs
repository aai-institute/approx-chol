//! AC vs AC2, whose `k` copies per edge trade more fill-in for lower factor variance.

#[path = "shared/grid.rs"]
mod grid;

use approx_chol::input::{Laplacian, Weights};
use approx_chol::{factorize, CliqueRule, Config, Multiplicity, Sgdd, SplitCount};
use grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(10, 10);
    println!("Grid Laplacian: 10×10 ({} vertices)", grid.n);

    let laplacian = Laplacian::new(grid.n, &grid.a, &grid.b, Weights::<f64>::unit())?;
    let sgdd: Sgdd = Sgdd::try_from(laplacian)?;
    let b = grid.split_rhs();

    let ac2 = Multiplicity::Split(SplitCount::new(2).expect("k ≥ 2"));
    let configs = [
        ("AC", Config::default()),
        (
            "AC2 (k=2)",
            Config {
                complement: CliqueRule::Sampled(ac2),
                approximate: ac2,
                ..Config::default()
            },
        ),
    ];
    for (name, config) in configs {
        let factor = factorize(&sgdd, &config);
        let (_, iterations) = grid.pcg(&factor, &b, 1e-10);
        println!("{name:>9}: {iterations} PCG iterations to 1e-10");
    }

    Ok(())
}
