//! Cross-tabulation of a view: the bipartite local Gramian.
//!
//! [`CrossTab`] holds `C` as a [`CsrBlock`] plus its transpose, built on first use
//! (rather than assembling the symmetric block matrix), and supports bipartite
//! connected-components splitting and per-component extraction.

use std::sync::OnceLock;

use num_traits::PrimInt;

use super::csr_block::{to_u32, CsrBlock};

mod accumulate;
use accumulate::accumulate_cross_block;
pub(crate) use accumulate::PairColumns;

/// A connected component in a bipartite factor-pair graph, in compact 0-based parent indices.
pub(crate) struct BipartiteComponent {
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
}

/// The off-diagonal block `C` of a channel pair's local Gramian, plus `Cᵀ`.
#[derive(Clone, Debug)]
pub(crate) struct CrossTab {
    /// CSR(C): row-block rows (n_rows) x col-block cols (n_cols).
    pub(crate) c: CsrBlock,
    /// CSR(Cᵀ), built on first use so a cover that only reads rows of `C` never pays for it.
    ct: OnceLock<CsrBlock>,
}

impl CrossTab {
    pub(crate) fn new(c: CsrBlock) -> Self {
        Self {
            c,
            ct: OnceLock::new(),
        }
    }

    pub(crate) fn with_transpose(c: CsrBlock, ct: CsrBlock) -> Self {
        debug_assert!(ct.nrows == c.ncols && ct.ncols == c.nrows);
        Self {
            c,
            ct: OnceLock::from(ct),
        }
    }

    /// Both blocks up front, for a cross-tab the solve will read on every iteration.
    pub(crate) fn eager(c: CsrBlock) -> Self {
        let ct = c.transpose();
        Self::with_transpose(c, ct)
    }

    pub(crate) fn ct(&self) -> &CsrBlock {
        self.ct.get_or_init(|| self.c.transpose())
    }

    pub(crate) fn into_parts(self) -> (CsrBlock, CsrBlock) {
        self.ct();
        let ct = self.ct.into_inner().expect("transpose was just built");
        (self.c, ct)
    }
}

impl CrossTab {
    /// Number of rows in the row block.
    pub(crate) fn n_rows(&self) -> usize {
        self.c.nrows
    }

    /// Number of rows in the col block.
    pub(crate) fn n_cols(&self) -> usize {
        self.c.ncols
    }

    /// Total number of DOFs (n_rows + n_cols).
    pub(crate) fn n_local(&self) -> usize {
        self.c.nrows + self.c.ncols
    }

    /// One view's `C`.
    pub(crate) fn build<I: PrimInt>(
        columns: PairColumns<'_, I, Option<&[f64]>, Option<&[f64]>>,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        CrossTab::eager(accumulate_cross_block(columns, n_rows, n_cols))
    }

    /// Symmetric adjacency over local `[q | r]` indexing: q-nodes walk `C`, r-nodes walk `Cᵀ`.
    pub(crate) fn neighbors(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let n_rows = self.n_rows();
        let (block, row, off) = if i < n_rows {
            (&self.c, i, n_rows)
        } else {
            (self.ct(), i - n_rows, 0)
        };
        block.row(row).map(move |(j, v)| (off + j, v))
    }

    /// Connected components by DFS over [`Self::neighbors`]; O(n_rows + n_cols + nnz).
    pub(crate) fn bipartite_connected_components(&self) -> Vec<BipartiteComponent> {
        let n_rows = self.n_rows();
        let mut visited = vec![false; self.n_local()];
        let mut components = Vec::new();
        let mut stack = Vec::new();

        for start in 0..self.n_local() {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            stack.push(start);
            let mut rows = Vec::new();
            let mut cols = Vec::new();

            while let Some(node) = stack.pop() {
                if node < n_rows {
                    rows.push(node);
                } else {
                    cols.push(node - n_rows);
                }
                for (j, _) in self.neighbors(node) {
                    if !visited[j] {
                        visited[j] = true;
                        stack.push(j);
                    }
                }
            }

            rows.sort_unstable();
            cols.sort_unstable();
            components.push(BipartiteComponent { rows, cols });
        }

        components
    }

    /// `row_remap`/`col_remap` must arrive all-`u32::MAX` and are reset on exit.
    pub(crate) fn extract_component(
        &self,
        comp: &BipartiteComponent,
        row_remap: &mut [u32],
        col_remap: &mut [u32],
    ) -> Self {
        let n_rows = comp.rows.len();
        let n_cols = comp.cols.len();
        debug_assert_eq!(row_remap.len(), self.n_rows());
        debug_assert_eq!(col_remap.len(), self.n_cols());

        for (new_idx, &old_idx) in comp.rows.iter().enumerate() {
            row_remap[old_idx] = to_u32(new_idx);
        }
        for (new_idx, &old_idx) in comp.cols.iter().enumerate() {
            col_remap[old_idx] = to_u32(new_idx);
        }

        let mut c_indptr = vec![0u32; n_rows + 1];
        let mut c_indices = Vec::new();
        let mut c_data = Vec::new();
        for (new_row, &old_row) in comp.rows.iter().enumerate() {
            let start = self.c.indptr[old_row] as usize;
            let end = self.c.indptr[old_row + 1] as usize;
            for idx in start..end {
                let old_rj = self.c.indices[idx] as usize;
                let new_rj = col_remap[old_rj];
                if new_rj != u32::MAX {
                    c_indices.push(new_rj);
                    c_data.push(self.c.data[idx]);
                }
            }
            c_indptr[new_row + 1] = to_u32(c_indices.len());
        }

        let c = CsrBlock {
            indptr: c_indptr,
            indices: c_indices,
            data: c_data,
            nrows: n_rows,
            ncols: n_cols,
        };
        for &old_idx in &comp.rows {
            row_remap[old_idx] = u32::MAX;
        }
        for &old_idx in &comp.cols {
            col_remap[old_idx] = u32::MAX;
        }

        CrossTab::eager(c)
    }
}
