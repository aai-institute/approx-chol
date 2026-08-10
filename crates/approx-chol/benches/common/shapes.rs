use approx_chol::OwnedCsr;

/// Connected canonical Laplacian where vertex `i` neighbours `i-bandwidth..=i+bandwidth`,
/// so row degree is `2 * bandwidth` independently of `n`.
pub fn banded_laplacian(n: usize, bandwidth: usize) -> OwnedCsr {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for vertex in 0..n {
        let low = vertex.saturating_sub(bandwidth);
        let high = (vertex + bandwidth).min(n - 1);
        let diagonal_slot = col_indices.len() + (vertex - low);
        let mut degree = 0.0f64;
        for other in low..=high {
            col_indices.push(other);
            if other == vertex {
                values.push(0.0);
            } else {
                values.push(-1.0);
                degree += 1.0;
            }
        }
        values[diagonal_slot] = degree;
        row_ptrs.push(col_indices.len());
    }
    OwnedCsr::try_from_usize(&row_ptrs, &col_indices, &values, n)
        .expect("banded laplacian must be valid CSR")
}

/// Row degree rises at fixed `n`, then `n` rises at fixed degree. Both matter: an
/// ingestion change can be 17% of the build at degree 4 and 0.2% at degree 511.
pub fn sweep() -> impl Iterator<Item = (String, OwnedCsr)> {
    [
        (1024usize, 2usize),
        (4608, 4),
        (1024, 16),
        (128, 128),
        (1024, 128),
        (512, 512),
    ]
    .into_iter()
    .map(|(n, bandwidth)| {
        let label = format!("n{n}_deg{}", (2 * bandwidth).min(n - 1));
        (label, banded_laplacian(n, bandwidth))
    })
}
