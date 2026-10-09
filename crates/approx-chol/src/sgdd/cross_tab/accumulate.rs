//! Observation accumulation kernels for [`CrossTab`](super::CrossTab) construction.

use num_traits::PrimInt;

use super::to_u32;
use super::CsrBlock;
use crate::sgdd::bipartite::Weights;

/// Hard cap on the dense accumulator (~40 MB); larger tables always go sparse.
const DENSE_TABLE_MAX_ENTRIES: usize = 5_000_000;

/// A channel's per-observation loading.
pub(crate) trait Loading: Copy {
    fn at(self, uid: usize) -> f64;
}

/// Intercept loading `l ≡ 1`; LLVM folds the resulting `x · 1.0` away.
#[derive(Clone, Copy)]
pub(crate) struct Unit;

impl Loading for Unit {
    #[inline]
    fn at(self, _uid: usize) -> f64 {
        1.0
    }
}

impl Loading for &[f64] {
    #[inline]
    fn at(self, uid: usize) -> f64 {
        self[uid]
    }
}

/// `None` is an intercept's `l ≡ 1`, for a reader that needs no specialized kernel.
impl Loading for Option<&[f64]> {
    #[inline]
    fn at(self, uid: usize) -> f64 {
        self.map_or(1.0, |z| z[uid])
    }
}

/// One weighted observation of a channel pair, at its cell with both loadings.
pub(crate) struct Observation {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) w: f64,
}

impl Observation {
    /// One expression for every reader: the forest's cancellation test needs bitwise-equal sums.
    #[inline]
    pub(crate) fn cell(&self) -> f64 {
        self.w * self.x * self.y
    }
}

/// Per-observation input columns backing one view: level codes, loadings, and weights.
#[derive(Clone, Copy)]
pub(crate) struct PairColumns<'a, I, Lq: Loading, Lr: Loading> {
    pub(crate) row_levels: &'a [I],
    pub(crate) col_levels: &'a [I],
    pub(crate) row_load: Lq,
    pub(crate) col_load: Lr,
    pub(crate) weights: Weights<'a>,
}

impl<'a, I: PrimInt, Lq: Loading, Lr: Loading> PairColumns<'a, I, Lq, Lr> {
    fn with_loadings<Mq: Loading, Mr: Loading>(
        self,
        row_load: Mq,
        col_load: Mr,
    ) -> PairColumns<'a, I, Mq, Mr> {
        PairColumns {
            row_levels: self.row_levels,
            col_levels: self.col_levels,
            row_load,
            col_load,
            weights: self.weights,
        }
    }

    pub(crate) fn n_obs(&self) -> usize {
        self.row_levels.len()
    }

    #[inline]
    pub(crate) fn observation(&self, uid: usize) -> Observation {
        Observation {
            row: self.row_levels[uid].to_usize().expect("row fits usize"),
            col: self.col_levels[uid].to_usize().expect("row fits usize"),
            x: self.row_load.at(uid),
            y: self.col_load.at(uid),
            w: self.weights.at(uid),
        }
    }
}

/// Accumulate into `C`, dispatching dense or sparse by peak transient memory.
pub(crate) fn accumulate_cross_block<I: PrimInt>(
    cols: PairColumns<'_, I, Option<&[f64]>, Option<&[f64]>>,
    n_rows: usize,
    n_cols: usize,
) -> CsrBlock {
    // Dispatching on cell count alone would pick sparse where it uses MORE memory.
    let table_size = n_rows.saturating_mul(n_cols);
    let dense_cost = table_size.saturating_mul(8);
    let sparse_cost = cols.n_obs().saturating_mul(12);
    let go_sparse = table_size > DENSE_TABLE_MAX_ENTRIES && sparse_cost < dense_cost;

    // One arm per loading combination, so each is monomorphized.
    match (cols.row_load, cols.col_load) {
        (None, None) => accumulate(cols.with_loadings(Unit, Unit), n_rows, n_cols, go_sparse),
        (Some(zq), None) => accumulate(cols.with_loadings(zq, Unit), n_rows, n_cols, go_sparse),
        (None, Some(zr)) => accumulate(cols.with_loadings(Unit, zr), n_rows, n_cols, go_sparse),
        (Some(zq), Some(zr)) => accumulate(cols.with_loadings(zq, zr), n_rows, n_cols, go_sparse),
    }
}

/// Size-dispatched accumulation for one monomorphized loading combination.
fn accumulate<I: PrimInt, Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, I, Lq, Lr>,
    n_rows: usize,
    n_cols: usize,
    go_sparse: bool,
) -> CsrBlock {
    if go_sparse {
        accumulate_sparse_cross_block(cols, n_rows, n_cols)
    } else {
        accumulate_dense_cross_block(cols, n_rows, n_cols)
    }
}

/// Dense path: flat `n_rows * n_cols` table with O(1) accumulation per observation.
fn accumulate_dense_cross_block<I: PrimInt, Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, I, Lq, Lr>,
    n_rows: usize,
    n_cols: usize,
) -> CsrBlock {
    let mut table = vec![0.0f64; n_rows * n_cols];

    for uid in 0..cols.n_obs() {
        let o = cols.observation(uid);
        debug_assert!(o.row < n_rows && o.col < n_cols);
        table[o.row * n_cols + o.col] += o.cell();
    }

    CsrBlock::from_dense_table(&table, n_rows, n_cols)
}

/// Sparse path: bucket by row, then dedup each row through a dense `n_cols` workspace.
fn accumulate_sparse_cross_block<I: PrimInt, Lq: Loading, Lr: Loading>(
    cols: PairColumns<'_, I, Lq, Lr>,
    n_rows: usize,
    n_cols: usize,
) -> CsrBlock {
    let n_obs = cols.n_obs();

    let mut row_counts = vec![0u32; n_rows];
    for uid in 0..n_obs {
        row_counts[cols.row_levels[uid].to_usize().expect("row fits usize")] += 1;
    }

    let mut bucket_indptr = vec![0u32; n_rows + 1];
    for i in 0..n_rows {
        bucket_indptr[i + 1] = bucket_indptr[i] + row_counts[i];
    }
    let total_entries = bucket_indptr[n_rows] as usize;

    let mut bucket_cols = vec![0u32; total_entries];
    let mut bucket_vals = vec![0.0f64; total_entries];
    let mut cursor = bucket_indptr[..n_rows].to_vec();
    for uid in 0..n_obs {
        let o = cols.observation(uid);
        let pos = cursor[o.row] as usize;
        bucket_cols[pos] = to_u32(o.col);
        bucket_vals[pos] = o.cell();
        cursor[o.row] += 1;
    }

    // A signed cell cancelling to 0.0 mid-row re-pushes its column; the duplicate is harmless.
    let mut work = vec![0.0f64; n_cols];
    let mut touched: Vec<u32> = Vec::new();
    let mut c_indptr = vec![0u32; n_rows + 1];
    let mut c_indices = Vec::new();
    let mut c_data = Vec::new();

    for row in 0..n_rows {
        let start = bucket_indptr[row] as usize;
        let end = bucket_indptr[row + 1] as usize;
        for idx in start..end {
            let col = bucket_cols[idx] as usize;
            if work[col] == 0.0 {
                touched.push(to_u32(col));
            }
            work[col] += bucket_vals[idx];
        }
        touched.sort_unstable();
        for &col in &touched {
            let v = work[col as usize];
            if v != 0.0 {
                c_indices.push(col);
                c_data.push(v);
            }
            work[col as usize] = 0.0;
        }
        c_indptr[row + 1] = to_u32(c_indices.len());
        touched.clear();
    }

    CsrBlock {
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
        nrows: n_rows,
        ncols: n_cols,
    }
}
