use crate::csr::OwnedCsr;

/// Unit-weight `rows × cols` grid Laplacian, row-major.
pub fn grid_laplacian(rows: usize, cols: usize) -> OwnedCsr {
    OwnedCsr::laplacian((0..rows * cols).map(|v| {
        let (r, c) = (v / cols, v % cols);
        [
            (r > 0).then(|| v - cols),
            (c > 0).then(|| v - 1),
            (c + 1 < cols).then_some(v + 1),
            (r + 1 < rows).then_some(v + cols),
        ]
        .into_iter()
        .flatten()
    }))
}
