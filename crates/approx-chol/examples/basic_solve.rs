//! Factorize a 10×10 grid Laplacian with AC and AC2, and use each to precondition CG.

use approx_chol::input::Laplacian;
use approx_chol::{factorize_with, Config, Factor, ScalingPolicy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Grid::new(10, 10);
    println!("Grid Laplacian: 10×10 ({} vertices)", grid.n);

    let laplacian = Laplacian::new(grid.n, &grid.a, &grid.b)?;
    let b = grid.split_rhs();

    // AC2's `k` copies per edge trade more fill-in for lower factor variance.
    let ac2 = Config {
        split_merge: Some(2),
        ..Config::default()
    };
    for (name, config) in [("AC", Config::default()), ("AC2 (k=2)", ac2)] {
        let factor = factorize_with(laplacian, config, ScalingPolicy::default())?;
        let (x, iterations) = grid.pcg(&factor, &b, 1e-10);
        let mean = x.iter().sum::<f64>() / grid.n as f64;
        let blocks = factor.blocks().len();
        println!(
            "{name:>9}: {blocks} block(s), {iterations} PCG iterations, mean {mean:.2e} (⟂ 1)"
        );
    }

    Ok(())
}

/// Unit-weight `rows × cols` grid, row-major, as the two ends of each edge.
struct Grid {
    n: usize,
    a: Vec<u32>,
    b: Vec<u32>,
}

impl Grid {
    fn new(rows: usize, cols: usize) -> Self {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for r in 0..rows {
            for c in 0..cols {
                let v = (r * cols + c) as u32;
                if c + 1 < cols {
                    a.push(v);
                    b.push(v + 1);
                }
                if r + 1 < rows {
                    a.push(v);
                    b.push(v + cols as u32);
                }
            }
        }
        Self {
            n: rows * cols,
            a,
            b,
        }
    }

    /// `+1` on the first half, `−1` on the second: zero-sum, so in the singular `L`'s range.
    fn split_rhs(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| if i < self.n / 2 { 1.0 } else { -1.0 })
            .collect()
    }

    /// `L x` straight from the edges, independent of the factor.
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len()];
        for (&a, &b) in self.a.iter().zip(&self.b) {
            let (a, b) = (a as usize, b as usize);
            let flow = x[a] - x[b];
            y[a] += flow;
            y[b] -= flow;
        }
        y
    }

    /// PCG on `L x = b` preconditioned by `factor`: the solution and its iteration count to `tol`.
    fn pcg(&self, factor: &Factor, b: &[f64], tol: f64) -> (Vec<f64>, usize) {
        let mut x = vec![0.0; b.len()];
        let mut r = b.to_vec();
        let mut z = factor.solve(&r).expect("solves");
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for iteration in 0..self.n {
            if dot(&r, &r).sqrt() <= tol * dot(b, b).sqrt() {
                return (x, iteration);
            }
            let ap = self.apply(&p);
            let alpha = rz / dot(&p, &ap);
            x.iter_mut().zip(&p).for_each(|(x, p)| *x += alpha * p);
            r.iter_mut().zip(&ap).for_each(|(r, ap)| *r -= alpha * ap);
            z = factor.solve(&r).expect("solves");
            let rz_next = dot(&r, &z);
            let beta = rz_next / rz;
            rz = rz_next;
            p.iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
        }
        (x, self.n)
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}
