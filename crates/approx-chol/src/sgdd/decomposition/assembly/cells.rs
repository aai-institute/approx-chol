//! `A`'s cells summed from `V`'s columns, row by row in column order.

use num_traits::PrimInt;

use crate::input::{Column, FactorWidth2};
use crate::sgdd::block::ReduceError;
use crate::storage::Rows;
use crate::{in_range, narrow, widen, Index, Real};

/// `A`'s strict upper triangle; `pair` sees each nonzero cell.
pub(super) fn upper<T: Real, J: PrimInt, I: Index, C: CellSum>(
    v: FactorWidth2<'_, T, J>,
    seen: impl FnMut(&Column<T, I>),
    pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let counts = counts::<T, J, I>(v, v.n(), seen);
    sum_cells::<T, J, I, C>(v, &counts, 0, pair)
}

/// The biadjacency `C` from the left part's rows, summed as [`Accumulator::of`] picks.
pub(super) fn cross<T: Real, J: PrimInt, I: Index, C: CellSum>(
    v: FactorWidth2<'_, T, J>,
    n_left: usize,
    seen: impl FnMut(&Column<T, I>),
    pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    Accumulator::of::<T, I, C>(n_left, v.n() - n_left, v.n_columns())
        .cross::<T, J, I, C>(v, n_left, seen, pair)
}

/// within's cap on the dense table's cells: about 40 MB plain, 80 MB valued.
const DENSE_TABLE_MAX_CELLS: usize = 5_000_000;

/// Measured crossover in instructions: the table wins once `V`'s columns reach a tenth of its cells.
const DENSE_TABLE_CELLS_PER_COLUMN: usize = 10;

/// Where `C`'s cells are summed: a dense table over both sides (+54% without it), or buckets per left row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Accumulator {
    Table,
    Buckets,
}

impl Accumulator {
    /// The table when columns fill it and it fits within's cap, or past the cap takes no more memory than the buckets.
    fn of<T, I, C>(n_left: usize, n_right: usize, n_columns: usize) -> Self {
        let cells = n_left.saturating_mul(n_right);
        let filled = n_columns.saturating_mul(DENSE_TABLE_CELLS_PER_COLUMN) >= cells;
        let fits = cells <= DENSE_TABLE_MAX_CELLS
            || Self::table_bytes::<T, I, C>(n_left, cells, n_columns)
                <= Self::bucket_bytes::<I, C>(n_left, n_right, n_columns);
        match filled && fits {
            true => Self::Table,
            false => Self::Buckets,
        }
    }

    /// The table and the rows built beside it: an offset per left row, a column and value per summed cell.
    fn table_bytes<T, I, C>(n_left: usize, cells: usize, n_columns: usize) -> usize {
        let per_cell = size_of::<I>() + size_of::<T>();
        (cells.saturating_mul(size_of::<C>()))
            .saturating_add(n_left.saturating_mul(size_of::<usize>()))
            .saturating_add(n_columns.min(cells).saturating_mul(per_cell))
    }

    /// Everything the buckets allocate: three offsets per left row, a column and cell per column, a sum and mark per right one.
    fn bucket_bytes<I, C>(n_left: usize, n_right: usize, n_columns: usize) -> usize {
        let per_left = 3 * size_of::<usize>();
        let per_column = size_of::<I>() + size_of::<f64>();
        let per_right = size_of::<C>() + size_of::<I>();
        (n_left.saturating_mul(per_left))
            .saturating_add(n_columns.saturating_mul(per_column))
            .saturating_add(n_right.saturating_mul(per_right))
    }

    fn cross<T: Real, J: PrimInt, I: Index, C: CellSum>(
        self,
        v: FactorWidth2<'_, T, J>,
        n_left: usize,
        seen: impl FnMut(&Column<T, I>),
        pair: impl FnMut(usize, usize, f64),
    ) -> Result<Rows<T, I>, ReduceError> {
        // `?` per arm: returning the arms' results directly reshuffled the whole build's inlining.
        let a = match self {
            Self::Table => dense_cross::<T, J, I, C>(v, n_left, seen, pair)?,
            Self::Buckets => {
                let counts = counts::<T, J, I>(v, n_left, seen);
                sum_cells::<T, J, I, C>(v, &counts, n_left, pair)?
            }
        };
        Ok(a)
    }
}

/// A cell's running sum over its columns' terms.
pub(in crate::sgdd) trait CellSum: Copy + Default {
    /// What `settled` needs to know of a cell's term count, counted over `V`'s columns.
    type Terms: Terms;

    fn add(&mut self, x: f64);

    fn sum(self) -> f64;

    /// The sum `A` keeps for the cell between vertices `a` and `b`.
    fn settled(self, terms: &Self::Terms, a: usize, b: usize) -> f64;
}

/// A count over `V`'s columns, each joining two vertices.
pub(in crate::sgdd) trait Terms {
    fn new(n: usize) -> Self;

    fn count(&mut self, a: usize, b: usize);
}

/// An incidence `V`'s terms share a sign, so no order cancels a cell to zero.
impl CellSum for f64 {
    type Terms = ();

    #[inline(always)]
    fn add(&mut self, x: f64) {
        *self += x;
    }

    #[inline(always)]
    fn sum(self) -> f64 {
        self
    }

    #[inline(always)]
    fn settled(self, _: &(), _: usize, _: usize) -> f64 {
        self
    }
}

impl Terms for () {
    fn new(_: usize) -> Self {}

    #[inline(always)]
    fn count(&mut self, _: usize, _: usize) {}
}

/// A valued `V`'s cell: its plain sum, and `Σ|x|` to bound how far another order's sum could be.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Bounded {
    sum: f64,
    magnitude: f64,
}

impl CellSum for Bounded {
    type Terms = PerVertex;

    #[inline(always)]
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.magnitude += x.abs();
    }

    #[inline(always)]
    fn sum(self) -> f64 {
        self.sum
    }

    /// Zero when within `k·ε·Σ|x|` of it, past any order's error `γ_(k−1)·Σ|x|` (ORO 2005, (2.4)).
    #[inline(always)]
    fn settled(self, terms: &PerVertex, a: usize, b: usize) -> f64 {
        // Empty cells skip the bound: on every table cell it cost +10% cycles.
        if self.sum == 0.0 {
            return 0.0;
        }
        let k = terms.0[a].min(terms.0[b]);
        // An overflowed `Σ|x|` bounds nothing; the plain sum goes on to `admit`.
        let off = self.magnitude * (k as f64 * f64::EPSILON);
        match off.is_finite() && self.sum.abs() <= off {
            true => 0.0,
            false => self.sum,
        }
    }
}

/// Each vertex's count of `V`'s terms, so a cell has at most its ends' smaller count.
pub(in crate::sgdd) struct PerVertex(Vec<usize>);

impl Terms for PerVertex {
    fn new(n: usize) -> Self {
        Self(vec![0; n])
    }

    #[inline(always)]
    fn count(&mut self, a: usize, b: usize) {
        self.0[a] += 1;
        self.0[b] += 1;
    }
}

/// Row `i`'s nonzero sum as `T` stores it; one cast to zero underflowed, so it is out of range too.
#[inline(always)]
fn admit<T: Real>(i: usize, x: f64) -> Result<T, ReduceError> {
    T::from(x)
        .filter(|&value| value != T::zero() && in_range(value))
        .ok_or(ReduceError::OutOfRange { vertex: i })
}

/// `pair` sees each nonzero cell.
fn dense_cross<T: Real, J: PrimInt, I: Index, C: CellSum>(
    v: FactorWidth2<'_, T, J>,
    n_left: usize,
    mut seen: impl FnMut(&Column<T, I>),
    mut pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let n_right = v.n() - n_left;
    let mut table = vec![C::default(); n_left * n_right];
    let mut terms = C::Terms::new(v.n());
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            table[column.a.index() * n_right + column.b.index() - n_left].add(column.cell());
            terms.count(column.a.index(), column.b.index());
            seen(&column);
        },
    );
    let mut ptrs = Vec::with_capacity(n_left + 1);
    ptrs.push(0);
    let (mut indices, mut cells) = (Vec::new(), Vec::new());
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        let sums =
            (row.iter().enumerate()).map(|(j, cell)| (j, cell.settled(&terms, i, n_left + j)));
        for (j, x) in sums.filter(|&(_, x)| x != 0.0) {
            pair(i, n_left + j, x);
            indices.push(I::from_index(j));
            cells.push(admit::<T>(i, x)?);
        }
        ptrs.push(indices.len());
    }
    // Freed before the shrinks reallocate; unshrunk, the factor kept the pushes' slack (+22% on a 5M AKM).
    drop(table);
    indices.shrink_to_fit();
    cells.shrink_to_fit();
    Ok(Rows::new(ptrs, indices, cells, n_right))
}

/// Each row's cell count, a column's row its lower end, `seen` visiting each column once.
fn counts<T: Real, J: PrimInt, I: Index>(
    v: FactorWidth2<'_, T, J>,
    n_rows: usize,
    mut seen: impl FnMut(&Column<T, I>),
) -> Vec<usize> {
    let mut counts = vec![0; n_rows];
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            counts[column.a.min(column.b).index()] += 1;
            seen(&column);
        },
    );
    counts
}

/// Cells bucketed at their lower end in column order, so each sums as a dense table would; `pair` sees each nonzero cell.
fn sum_cells<T: Real, J: PrimInt, I: Index, C: CellSum>(
    v: FactorWidth2<'_, T, J>,
    counts: &[usize],
    first_column: usize,
    mut pair: impl FnMut(usize, usize, f64),
) -> Result<Rows<T, I>, ReduceError> {
    let starts: Vec<usize> = std::iter::once(0)
        .chain(counts.iter().scan(0, |end, &count| {
            *end += count;
            Some(*end)
        }))
        .collect();
    let entries = starts[counts.len()];
    let mut cursor = starts[..counts.len()].to_vec();
    let (mut columns, mut cells) = (vec![I::zero(); entries], vec![0.0; entries]);
    let mut terms = C::Terms::new(v.n());
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            let (lo, hi) = (column.a.min(column.b), column.a.max(column.b));
            terms.count(lo.index(), hi.index());
            let slot = &mut cursor[lo.index()];
            columns[*slot] = hi - I::from_index(first_column);
            cells[*slot] = column.cell();
            *slot += 1;
        },
    );
    drop(cursor);
    let n_cols = v.n() - first_column;
    let mut sums = vec![C::default(); n_cols];
    let mut touched: Vec<I> = Vec::new();
    let mut ptrs = Vec::with_capacity(starts.len());
    ptrs.push(0);
    let mut out = 0;
    for (i, span) in starts.windows(2).enumerate() {
        for p in span[0]..span[1] {
            let j = columns[p];
            // A cell cancelling to zero mid-row is pushed again; the dedup absorbs it.
            if sums[j.index()].sum() == 0.0 {
                touched.push(j);
            }
            sums[j.index()].add(cells[p]);
        }
        touched.sort_unstable();
        touched.dedup();
        // Written in place: a row's sums never outnumber its own bucket, already read.
        for &j in &touched {
            let cell =
                std::mem::take(&mut sums[j.index()]).settled(&terms, i, first_column + j.index());
            if cell != 0.0 {
                pair(i, first_column + j.index(), cell);
                columns[out] = j;
                cells[out] = widen(admit::<T>(i, cell)?);
                out += 1;
            }
        }
        touched.clear();
        ptrs.push(out);
    }
    // Freed before the shrinks reallocate, as they set the reduction's peak.
    drop((starts, sums, terms));
    columns.truncate(out);
    columns.shrink_to_fit();
    cells.truncate(out);
    cells.shrink_to_fit();
    let values = cells.into_iter().map(narrow).collect();
    Ok(Rows::new(ptrs, columns, values, n_cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{BipartiteFactorWidth2, FactorWidth1};

    fn filling(cells: usize) -> usize {
        cells.div_ceil(DENSE_TABLE_CELLS_PER_COLUMN)
    }

    #[test]
    fn below_the_cap_the_table_needs_only_fill() {
        let (n_left, n_right) = (20, 10);
        let n_columns = filling(n_left * n_right);
        let of = |n_columns| Accumulator::of::<f64, u32, f64>(n_left, n_right, n_columns);
        assert_eq!(of(n_columns), Accumulator::Table);
        assert_eq!(of(n_columns - 1), Accumulator::Buckets);
    }

    #[test]
    fn past_the_cap_the_table_also_needs_less_memory_than_the_buckets() {
        let (n_left, n_right) = (2000, DENSE_TABLE_MAX_CELLS / 2000);
        let of =
            |n_right| Accumulator::of::<f64, u32, f64>(n_left, n_right, filling(n_left * n_right));
        assert_eq!(of(n_right), Accumulator::Table);
        assert_eq!(of(n_right + 1), Accumulator::Buckets);
        // 9M cells under 14M repeated columns: 12-byte buckets stay below the table and its rows, 16-byte ones do not.
        assert_eq!(
            Accumulator::of::<f64, u32, f64>(3000, 3000, 14_000_000),
            Accumulator::Buckets
        );
        assert_eq!(
            Accumulator::of::<f64, u64, f64>(3000, 3000, 14_000_000),
            Accumulator::Table
        );
        // A valued table's 16-byte cells tip it past the 16-byte buckets.
        assert_eq!(
            Accumulator::of::<f64, u64, Bounded>(3000, 3000, 14_000_000),
            Accumulator::Buckets
        );
        // One right vertex: the table and its rows take 16 bytes per left row, the buckets' offsets alone 24.
        let n_left = DENSE_TABLE_MAX_CELLS + 1;
        assert_eq!(
            Accumulator::of::<f64, u32, f64>(n_left, 1, filling(n_left)),
            Accumulator::Table
        );
        // Six right vertices, four columns a row: 48 MB of table, but 104 MB with its rows against 72 MB of buckets.
        assert_eq!(
            Accumulator::of::<f64, u32, f64>(1_000_000, 6, 4_000_000),
            Accumulator::Buckets
        );
    }

    type Summed = (String, Vec<(u32, u32, f64)>, Vec<(usize, usize, f64)>);

    fn sum<C: CellSum>(
        accumulator: Accumulator,
        gram: BipartiteFactorWidth2<'_, f64>,
        n_left: usize,
    ) -> Summed {
        let (mut seen, mut pairs) = (Vec::new(), Vec::new());
        let rows = accumulator.cross::<f64, u32, u32, C>(
            gram.0,
            n_left,
            |column| seen.push((column.a, column.b, column.cell())),
            |i, j, cell| pairs.push((i, j, cell)),
        );
        (format!("{:?}", rows.unwrap()), seen, pairs)
    }

    #[test]
    fn table_and_buckets_sum_valued_and_incidence_sides_alike() {
        let left = [0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9];
        let right = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 0, 1];
        let x = [
            1.0, 2.0, 1.5, 1.0, 0.5, 1.25, 2.0, 2.0, 1.0, 3.0, 1.0, 0.75, 1.0, 2.5, 1.0,
        ];
        let y = [
            -1.0, 0.25, -2.0, 1.5, -1.5, 1.0, 0.75, -0.75, -1.0, 1.0, -0.5, 2.0, 1.0, -1.0, 0.25,
        ];
        let s = [
            1.0, 2.0, 0.5, 1.0, 1.0, 0.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let n_left = 20;
        let valued = BipartiteFactorWidth2::new(
            FactorWidth1::new(n_left, &left, Some(&x)).unwrap(),
            FactorWidth1::new(10, &right, Some(&y)).unwrap(),
        );
        let incidence = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(n_left, &left, None).unwrap(),
            FactorWidth1::new(10, &right, None).unwrap(),
            &s,
        );
        let (valued, incidence) = (valued.unwrap(), incidence.unwrap());
        assert_eq!(
            sum::<Bounded>(Accumulator::Table, valued, n_left),
            sum::<Bounded>(Accumulator::Buckets, valued, n_left)
        );
        assert_eq!(
            sum::<f64>(Accumulator::Table, incidence, n_left),
            sum::<f64>(Accumulator::Buckets, incidence, n_left)
        );
    }

    fn p(e: i32) -> f64 {
        2f64.powi(e)
    }

    /// Cell (0, 0) sums to 1, below its terms' rounding: plainly `−2⁵⁶` in the first order, 1 in the second.
    #[rstest::rstest]
    #[case::one_amid([p(110), p(56), -p(110), p(53), 1.0, -p(53), -p(56)])]
    #[case::one_last([p(110), -p(110), p(56), -p(56), p(53), -p(53), 1.0])]
    fn table_and_buckets_drop_a_cell_below_its_rounding(#[case] terms: [f64; 7]) {
        let x = [&terms[..], &[1.0]].concat();
        let rows = [0u32, 0, 0, 0, 0, 0, 0, 1];
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &rows, Some(&x[..])).unwrap(),
            FactorWidth1::new(2, &rows, None).unwrap(),
        )
        .unwrap();
        for accumulator in [Accumulator::Table, Accumulator::Buckets] {
            let pairs = sum::<Bounded>(accumulator, gram, 2).2;
            assert_eq!(pairs, [(1, 3, 1.0)], "{accumulator:?}");
        }
    }

    /// Cell (0, 0) sums exactly to 2⁻⁴⁰ beside 3000 columns elsewhere, which a bound by `V`'s count drops.
    #[test]
    fn table_and_buckets_bound_a_cell_by_its_own_terms() {
        let x = [&[1.0, -1.0 + p(-40)][..], &[1.0; 3000]].concat();
        let rows = [vec![0u32; 2], vec![1; 3000]].concat();
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &rows, Some(&x[..])).unwrap(),
            FactorWidth1::new(2, &rows, None).unwrap(),
        )
        .unwrap();
        for accumulator in [Accumulator::Table, Accumulator::Buckets] {
            let pairs = sum::<Bounded>(accumulator, gram, 2).2;
            assert_eq!(pairs, [(0, 2, p(-40)), (1, 3, 3000.0)], "{accumulator:?}");
        }
    }

    /// Eleven cells: pushes doubling past them leave slack the factor would keep.
    #[test]
    fn a_dense_cross_table_keeps_no_slack() {
        let left: Vec<u32> = (0..11).map(|k| k % 4).collect();
        let right: Vec<u32> = (0..11).collect();
        let x = [1.0; 11];
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(4, &left, Some(&x)).unwrap(),
            FactorWidth1::new(11, &right, Some(&x)).unwrap(),
        )
        .unwrap();
        let rows = Accumulator::Table
            .cross::<f64, u32, u32, Bounded>(gram.0, 4, |_| {}, |_, _, _| {})
            .unwrap();
        assert!(rows.holds_no_slack());
    }
}
