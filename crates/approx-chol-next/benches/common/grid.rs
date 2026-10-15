use crate::csr::OwnedCsr;

/// Unit-weight `rows × cols` grid Laplacian, row-major.
pub fn grid_laplacian(rows: usize, cols: usize) -> OwnedCsr {
    OwnedCsr::from_rows((0..rows * cols).map(|v| {
        let (r, c) = (v / cols, v % cols);
        let mut row = Vec::with_capacity(5);
        if r > 0 {
            row.push((v - cols, -1.0));
        }
        if c > 0 {
            row.push((v - 1, -1.0));
        }
        let diagonal = row.len();
        row.push((v, 0.0));
        if c + 1 < cols {
            row.push((v + 1, -1.0));
        }
        if r + 1 < rows {
            row.push((v + cols, -1.0));
        }
        row[diagonal].1 = (row.len() - 1) as f64;
        row
    }))
}
