use approx_chol_next::input::Csr;
use approx_chol_next::{ScalingPolicy, Sgdd};

/// A symmetric `A` held as owned CSR arrays, as main's benches hand it over.
pub struct OwnedCsr {
    pub n: usize,
    pub row_ptrs: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl OwnedCsr {
    /// Rows of `(column, value)` in column order, the diagonal included.
    pub fn from_rows(rows: impl IntoIterator<Item = Vec<(usize, f64)>>) -> Self {
        let (mut row_ptrs, mut col_indices, mut values) = (vec![0u32], Vec::new(), Vec::new());
        for row in rows {
            for (col, value) in row {
                col_indices.push(col as u32);
                values.push(value);
            }
            row_ptrs.push(col_indices.len() as u32);
        }
        Self {
            n: row_ptrs.len() - 1,
            row_ptrs,
            col_indices,
            values,
        }
    }

    pub fn csr(&self) -> Csr<'_, f64> {
        Csr::new(self.n, &self.row_ptrs, &self.col_indices, &self.values).expect("valid CSR")
    }
}

/// What main's `Builder::build` did before eliminating: check symmetry and reduce.
pub fn reduce(csr: Csr<'_, f64>) -> Sgdd {
    Sgdd::from_csr(csr, ScalingPolicy::default()).expect("reduces")
}
