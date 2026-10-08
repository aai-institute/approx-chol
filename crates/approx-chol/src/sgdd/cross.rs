//! The off-diagonal block `C` of a bipartite matrix, rows by columns, with its transpose.

use super::bipartite::{Gram, Graph};
use num_traits::PrimInt;

/// Row-major sparse block with ascending columns per row.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub(crate) indptr: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) data: Vec<f64>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl Block {
    pub(crate) fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn span(&self, i: usize) -> core::ops::Range<usize> {
        self.indptr[i] as usize..self.indptr[i + 1] as usize
    }

    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let span = self.span(i);
        (self.indices[span.clone()].iter())
            .zip(&self.data[span])
            .map(|(&j, &w)| (j as usize, w))
    }

    /// Output rows come out sorted because source rows go ascending.
    pub(crate) fn transpose(&self) -> Block {
        let mut indptr = vec![0u32; self.ncols + 1];
        for &col in &self.indices {
            indptr[col as usize + 1] += 1;
        }
        for i in 0..self.ncols {
            indptr[i + 1] += indptr[i];
        }
        let mut cursor = indptr[..self.ncols].to_vec();
        let mut indices = vec![0u32; self.nnz()];
        let mut data = vec![0.0f64; self.nnz()];
        for src in 0..self.nrows {
            for (j, value) in self.row(src) {
                let pos = cursor[j] as usize;
                indices[pos] = src as u32;
                data[pos] = value;
                cursor[j] += 1;
            }
        }
        Block {
            indptr,
            indices,
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// `y = base + self · x`, accumulated from `base` as within does.
    pub(crate) fn multiply_add(&self, x: &[f64], base: &[f64], y: &mut [f64]) {
        for (i, out) in y[..self.nrows].iter_mut().enumerate() {
            let mut acc = base[i];
            for (j, v) in self.row(i) {
                acc += v * x[j];
            }
            *out = acc;
        }
    }
}

/// `C` with `Cᵀ`, over local `[rows | cols]` numbering.
#[derive(Clone, Debug)]
pub(crate) struct Cross {
    pub(crate) c: Block,
    pub(crate) ct: Block,
}

impl Cross {
    pub(crate) fn new(c: Block) -> Self {
        let ct = c.transpose();
        Self { c, ct }
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.c.nrows
    }

    pub(crate) fn n_cols(&self) -> usize {
        self.c.ncols
    }

    pub(crate) fn n_local(&self) -> usize {
        self.c.nrows + self.c.ncols
    }

    /// Rows walk `C`, columns walk `Cᵀ`.
    pub(crate) fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let n_rows = self.n_rows();
        let (block, row, offset) = if i < n_rows {
            (&self.c, i, n_rows)
        } else {
            (&self.ct, i - n_rows, 0)
        };
        block.row(row).map(move |(j, v)| (offset + j, v))
    }

    /// Swaps the roles, so the larger side comes first.
    pub(crate) fn swapped(self) -> Self {
        Self {
            c: self.ct,
            ct: self.c,
        }
    }

    /// `rows` and `cols` ascending; `remap` arrives and leaves all `u32::MAX`, sized `n_cols`.
    pub(crate) fn extract(&self, rows: &[usize], cols: &[usize], remap: &mut [u32]) -> Self {
        for (local, &col) in cols.iter().enumerate() {
            remap[col] = local as u32;
        }
        let mut indptr = Vec::with_capacity(rows.len() + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for &row in rows {
            for (j, value) in self.c.row(row) {
                if remap[j] != u32::MAX {
                    indices.push(remap[j]);
                    data.push(value);
                }
            }
            indptr.push(indices.len() as u32);
        }
        for &col in cols {
            remap[col] = u32::MAX;
        }
        Self::new(Block {
            indptr,
            indices,
            data,
            nrows: rows.len(),
            ncols: cols.len(),
        })
    }
}

/// Buckets by row in column order, then sums each row's cells through a dense workspace.
fn accumulate(n_rows: usize, n_cols: usize, cells: &[(u32, u32, f64)]) -> Block {
    let mut bucket_ptr = vec![0u32; n_rows + 1];
    for &(row, _, _) in cells {
        bucket_ptr[row as usize + 1] += 1;
    }
    for i in 0..n_rows {
        bucket_ptr[i + 1] += bucket_ptr[i];
    }
    let mut cursor = bucket_ptr[..n_rows].to_vec();
    let mut bucket = vec![(0u32, 0.0f64); cells.len()];
    for &(row, col, cell) in cells {
        bucket[cursor[row as usize] as usize] = (col, cell);
        cursor[row as usize] += 1;
    }

    let mut work = vec![0.0f64; n_cols];
    let mut touched: Vec<u32> = Vec::new();
    let mut indptr = vec![0u32; n_rows + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..n_rows {
        for &(col, cell) in &bucket[bucket_ptr[row] as usize..bucket_ptr[row + 1] as usize] {
            if work[col as usize] == 0.0 {
                touched.push(col);
            }
            work[col as usize] += cell;
        }
        touched.sort_unstable();
        touched.dedup();
        for &col in &touched {
            let value = work[col as usize];
            if value != 0.0 {
                indices.push(col);
                data.push(value);
            }
            work[col as usize] = 0.0;
        }
        indptr[row + 1] = indices.len() as u32;
        touched.clear();
    }
    Block {
        indptr,
        indices,
        data,
        nrows: n_rows,
        ncols: n_cols,
    }
}

/// A graph's `C` is its summed weights; its diagonal is `C`'s row sums, since its surplus is zero.
pub(crate) fn of_graph<I: PrimInt>(graph: &Graph<'_, I>) -> Cross {
    let (n_left, n_right) = graph.sizes();
    let cells: Vec<(u32, u32, f64)> = (graph.columns())
        .map(|(l, r, w)| (l as u32, r as u32, w))
        .collect();
    Cross::new(accumulate(n_left, n_right, &cells))
}

/// A Gram's `C` and its diagonal, each summed in column order.
pub(crate) fn of_gram<I: PrimInt>(gram: &Gram<'_, I>) -> (Cross, Vec<f64>) {
    let (n_left, n_right) = gram.sizes();
    let mut diagonal = vec![0.0f64; n_left + n_right];
    let mut cells = Vec::with_capacity(gram.n_columns());
    for k in 0..gram.n_columns() {
        let o = gram.observation(k);
        diagonal[o.row] += o.w * o.x * o.x;
        diagonal[n_left + o.col] += o.w * o.y * o.y;
        cells.push((o.row as u32, o.col as u32, o.cell()));
    }
    (Cross::new(accumulate(n_left, n_right, &cells)), diagonal)
}
