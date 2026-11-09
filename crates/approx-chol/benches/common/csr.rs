use approx_chol::input::{CsrRef, Input, TriangleCsrRef};
use approx_chol::{factorize_with, ApproxInverse, Config, ScalingPolicy};

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

    /// `D L D` for the weighted `edges`, a repeated pair summed once so both triangles hold the same bits: off-diagonal `−w_ij (d_i d_j)`, diagonal `d_i² Σ_j w_ij`.
    #[allow(dead_code, reason = "only the reduce bench scales a Laplacian")]
    pub fn congruent_laplacian(
        d: &[f64],
        edges: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Self {
        let mut edges: Vec<_> = (edges.into_iter())
            .map(|(a, b, w)| (a.min(b), a.max(b), w))
            .collect();
        edges.sort_unstable_by_key(|&(a, b, _)| (a, b));
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d.len()];
        let mut edges = edges.into_iter().peekable();
        while let Some((a, b, mut w)) = edges.next() {
            while let Some((_, _, more)) = edges.next_if(|&(c, e, _)| (c, e) == (a, b)) {
                w += more;
            }
            rows[a].push((b, w));
            rows[b].push((a, w));
        }
        Self::from_rows(rows.into_iter().enumerate().map(|(i, mut row)| {
            let degree: f64 = row.iter().map(|&(_, w)| w).sum();
            row.push((i, 0.0));
            row.sort_unstable_by_key(|&(j, _)| j);
            row.into_iter().map(move |(j, w)| match j == i {
                true => (j as u32, d[i] * d[i] * degree),
                false => (j as u32, -(w * (d[i] * d[j]))),
            })
        }))
    }

    pub fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    /// Row `i`'s stored entries `(column, value)`, in the order given.
    #[allow(
        dead_code,
        reason = "only the ingest and frustrated benches read rows or apply A"
    )]
    pub fn row(&self, i: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let span = self.row_ptrs[i] as usize..self.row_ptrs[i + 1] as usize;
        (self.col_indices[span.clone()].iter().copied()).zip(self.values[span].iter().copied())
    }

    /// Stored entries, both triangles.
    #[allow(dead_code, reason = "only the frustrated benches count entries")]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn csr(&self) -> CsrRef<'_, f64> {
        CsrRef::new(self.n(), &self.row_ptrs, &self.col_indices, &self.values).expect("valid CSR")
    }

    /// `A x` over every stored entry.
    #[allow(dead_code, reason = "only the frustrated benches apply A")]
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n())
            .map(|i| self.row(i).map(|(j, a)| a * x[j as usize]).sum())
            .collect()
    }

    /// The arrays read as an upper triangle, diagonal included.
    #[allow(dead_code, reason = "only the ingest bench stores a triangle")]
    pub fn upper(&self) -> TriangleCsrRef<'_, f64> {
        TriangleCsrRef::upper(self.n(), &self.row_ptrs, &self.col_indices, &self.values)
            .expect("valid CSR")
    }
}

/// Input checks and reduction belong to the timed build: every caller pays them.
pub fn build<'a, M>(input: M, config: Config) -> ApproxInverse
where
    M: Into<Input<'a, f64, u32>>,
{
    factorize_with(input, config, ScalingPolicy::default()).expect("factors")
}
