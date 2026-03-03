use approx_chol::CsrRef;

/// Grid Laplacian stored as owned arrays (CsrRef-compatible).
pub struct GridLaplacian {
    pub row_ptrs: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
    pub n: u32,
}

impl GridLaplacian {
    pub fn as_csr(&self) -> CsrRef<'_> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n)
            .expect("grid_laplacian must build valid CSR")
    }
}

/// Build a 2D grid Laplacian of size `rows x cols`.
pub fn grid_laplacian(rows: usize, cols: usize) -> GridLaplacian {
    let n = rows * cols;
    let mut row_ptrs: Vec<u32> = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);

    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            let mut diag = 0.0f64;
            let mut neighbors = Vec::new();

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
