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

use approx_chol::{Builder, Config, CsrRef};

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
    fn as_csr(&self) -> CsrRef<'_> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n)
            .expect("grid_laplacian must build valid CSR")
    }
}

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

fn main() {
    let lap = grid_laplacian(10, 10);
    let n = lap.n as usize;
    println!("Grid Laplacian: 10×10 ({n} nodes)\n");

    // -----------------------------------------------------------------------
    // Default AC (split_merge = None)
    // -----------------------------------------------------------------------
    let ac_config = Config::default();
    let ac_factor = Builder::new(ac_config)
        .build(lap.as_csr())
        .expect("AC factorization failed");

    println!("=== Default AC ===");
    println!("  split_merge : None (standard AC)");
    println!("  n_steps     : {}", ac_factor.n_steps());
    println!("  factor dim  : {}", ac_factor.n());

    // -----------------------------------------------------------------------
    // AC2 (k = 2)
    // -----------------------------------------------------------------------
    let ac2_config = Config {
        split_merge: Some(2),
        seed: 42,
    };
    let ac2_factor = Builder::new(ac2_config)
        .build(lap.as_csr())
        .expect("AC2 factorization failed");

    println!("\n=== AC2 (k=2) ===");
    println!("  split_merge : Some(2)");
    println!("  n_steps     : {}", ac2_factor.n_steps());
    println!("  factor dim  : {}", ac2_factor.n());

    // The factor dimension is the same — AC2 doesn't augment the matrix,
    // it only changes how edges are sampled during factorization.
    assert_eq!(
        ac_factor.n(),
        ac2_factor.n(),
        "AC and AC2 must have the same factor dimension"
    );
    println!(
        "\nNote: factor dimensions match ({}) — AC2 changes sampling quality, not matrix size.",
        ac_factor.n()
    );

    // -----------------------------------------------------------------------
    // Solve the same system with both factors and compare
    // -----------------------------------------------------------------------
    let mut b = vec![0.0f64; n];
    for (i, bi) in b.iter_mut().enumerate() {
        *bi = if i < n / 2 { 1.0 } else { -1.0 };
    }

    let x_ac = ac_factor.solve(&b).expect("AC solve failed");
    let x_ac2 = ac2_factor.solve(&b).expect("AC2 solve failed");

    let norm_ac: f64 = x_ac.iter().map(|v| v * v).sum::<f64>().sqrt();
    let norm_ac2: f64 = x_ac2.iter().map(|v| v * v).sum::<f64>().sqrt();

    println!("\nSolution norms:");
    println!("  AC  : {norm_ac:.6}");
    println!("  AC2 : {norm_ac2:.6}");
    println!("  (norms differ — approximate factors have different spectral quality)");
}
