//! Factorize a 10×10 grid Laplacian and use it to precondition CG.

#[path = "shared/grid.rs"]
mod grid;

use approx_chol::factorize;
use approx_chol::input::{Laplacian, Weights};
use grid::Grid;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(10, 10);
    println!("Grid Laplacian: 10×10 ({} vertices)", grid.n);

    let laplacian = Laplacian::new(grid.n, &grid.a, &grid.b, Weights::<f64>::unit())?;
    let factor = factorize(laplacian)?;
    println!("Irreducible blocks: {}", factor.blocks().len());
    let (x, iterations) = grid.pcg(&factor, &grid.split_rhs(), 1e-10);

    let mean = x.iter().sum::<f64>() / grid.n as f64;
    println!("PCG: {iterations} iterations to 1e-10");
    println!("Solution mean: {mean:.2e}  (the floating Laplacian's solve is orthogonal to 1)");

    Ok(())
}
