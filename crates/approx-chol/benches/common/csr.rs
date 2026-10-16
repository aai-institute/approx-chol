use approx_chol::input::CsrRef;
use approx_chol::{factorize_with, Config, Factor};

/// A unit-weight graph Laplacian held as owned u32 CSR arrays.
pub struct OwnedCsr {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
}

impl OwnedCsr {
    /// Each vertex's neighbours in increasing order, itself excluded.
    pub fn laplacian(neighbors: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Self {
        let (mut row_ptrs, mut col_indices, mut values) = (vec![0u32], Vec::new(), Vec::new());
        for (vertex, row) in neighbors.into_iter().enumerate() {
            let start = col_indices.len();
            col_indices.extend(row.into_iter().map(|u| u as u32));
            let degree = col_indices.len() - start;
            values.resize(col_indices.len(), -1.0);
            let diagonal = start + col_indices[start..].partition_point(|&u| (u as usize) < vertex);
            col_indices.insert(diagonal, vertex as u32);
            values.insert(diagonal, degree as f64);
            row_ptrs.push(col_indices.len() as u32);
        }
        Self {
            row_ptrs,
            col_indices,
            values,
        }
    }

    pub fn csr(&self) -> CsrRef<'_, f64> {
        let n = (self.row_ptrs.len() - 1) as u32;
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, n).expect("valid CSR")
    }
}

/// Symmetry check and reduction belong to the timed build: every caller pays them.
pub fn build(csr: CsrRef<'_, f64>, config: Config) -> Factor {
    factorize_with(csr, config).expect("factors")
}
