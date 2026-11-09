use approx_chol::input::{CsrRef, UpperCsrRef};
use approx_chol::{factorize_with, ApproxInverse, Config, Error, Input, ScalingPolicy};

/// CSR arrays as a caller holds them, `u32` indices and `f64` values, each row's entries in the order given.
pub struct OwnedCsr {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
}

impl OwnedCsr {
    /// Row `i`'s entries `(column, value)` as the `i`-th item.
    pub fn from_rows(rows: impl IntoIterator<Item = impl IntoIterator<Item = (u32, f64)>>) -> Self {
        let (mut row_ptrs, mut col_indices, mut values) = (vec![0u32], Vec::new(), Vec::new());
        for row in rows {
            for (j, a) in row {
                col_indices.push(j);
                values.push(a);
            }
            row_ptrs.push(col_indices.len() as u32);
        }
        Self {
            row_ptrs,
            col_indices,
            values,
        }
    }

    /// The unit-weight graph Laplacian of each vertex's neighbours in increasing order, itself excluded.
    pub fn laplacian(neighbors: impl IntoIterator<Item = impl IntoIterator<Item = usize>>) -> Self {
        Self::from_rows(neighbors.into_iter().enumerate().map(|(vertex, row)| {
            let row: Vec<u32> = row.into_iter().map(|u| u as u32).collect();
            let diagonal = row.partition_point(|&u| (u as usize) < vertex);
            let (below, above) = row.split_at(diagonal);
            let off = |&u: &u32| (u, -1.0);
            (below.iter().map(off))
                .chain([(vertex as u32, row.len() as f64)])
                .chain(above.iter().map(off))
                .collect::<Vec<_>>()
        }))
    }

    fn n(&self) -> u32 {
        (self.row_ptrs.len() - 1) as u32
    }

    pub fn csr(&self) -> CsrRef<'_, f64> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n()).expect("valid CSR")
    }

    /// The arrays read as an upper triangle, diagonal included.
    #[allow(dead_code, reason = "only the ingest bench stores a triangle")]
    pub fn upper(&self) -> UpperCsrRef<'_, f64> {
        UpperCsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n())
            .expect("valid CSR")
    }
}

/// Input checks and reduction belong to the timed build: every caller pays them.
pub fn build<'a, M>(input: M, config: Config) -> ApproxInverse
where
    M: TryInto<Input<'a, f64, u32>>,
    M::Error: Into<Error>,
{
    factorize_with(input, config, ScalingPolicy::default()).expect("factors")
}
