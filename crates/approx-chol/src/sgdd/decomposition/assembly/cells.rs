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

/// within's cap on the dense table, in plain cells: about 40 MB, which a valued table's wider cells share.
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
        let fits = cells.saturating_mul(size_of::<C>()) <= DENSE_TABLE_MAX_CELLS * size_of::<f64>()
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
    /// Its terms can take both signs, so their order could cancel it to zero.
    const CAN_CANCEL: bool = false;

    fn add(&mut self, x: f64);

    fn sum(self) -> f64;

    /// Its plain sum is within `slack·Σ|x|` of zero, `slack` bounding any order's rounding.
    fn needs_exact(self, _slack: f64) -> bool {
        false
    }
}

/// An incidence `V`'s terms share a sign, so no order cancels a cell to zero.
impl CellSum for f64 {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        *self += x;
    }

    fn sum(self) -> f64 {
        self
    }
}

/// A valued `V`'s cell: its plain sum, and `Σ|x|` to bound how far another order's sum could be.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Bounded {
    sum: f64,
    magnitude: f64,
}

impl CellSum for Bounded {
    const CAN_CANCEL: bool = true;

    #[inline(always)]
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.magnitude += x.abs();
    }

    fn sum(self) -> f64 {
        self.sum
    }

    /// Any order's plain sum of `k` terms is off by at most `γ_(k−1)·Σ|x|` (ORO 2005, (2.4)), below `k·ε·Σ|x|`.
    #[inline(always)]
    fn needs_exact(self, slack: f64) -> bool {
        // `off` is zero when empty or all subnormal, where sums are exact; NaN or ∞ ends at `admit`.
        let off = self.magnitude * slack;
        0.0 < off && self.sum.abs() <= off
    }
}

/// `Σ terms` exactly, by Shewchuk's Grow-Expansion (DCG 18(3), 1997, Thm 10): zero only if the sum is.
#[cold]
#[inline(never)]
fn exact_sum(terms: impl IntoIterator<Item = f64>) -> f64 {
    let mut parts: Vec<f64> = Vec::new();
    for mut x in terms {
        let mut kept = 0;
        for k in 0..parts.len() {
            let error;
            (x, error) = crate::two_sum(x, parts[k]);
            if error != 0.0 {
                parts[kept] = error;
                kept += 1;
            }
        }
        parts.truncate(kept);
        if x != 0.0 {
            parts.push(x);
        }
    }
    // Nonoverlapping and increasing, so the top part dominates and the sum cannot round to zero.
    parts.iter().sum()
}

/// The exact sums of the cells in `terms`, `(cell, term)` pairs, in cell order.
#[cold]
#[inline(never)]
fn exact_sums(mut terms: Vec<(usize, f64)>) -> std::vec::IntoIter<f64> {
    terms.sort_unstable_by_key(|&(k, _)| k);
    let cells = terms.chunk_by(|a, b| a.0 == b.0);
    let sums: Vec<f64> = cells
        .map(|cell| exact_sum(cell.iter().map(|&(_, x)| x)))
        .collect();
    sums.into_iter()
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
    let at = |column: &Column<T, I>| column.a.index() * n_right + column.b.index() - n_left;
    let mut table = vec![C::default(); n_left * n_right];
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            table[at(&column)].add(column.cell());
            seen(&column);
        },
    );
    let slack = v.n_columns() as f64 * f64::EPSILON;
    let mut exact = None;
    let mut ptrs = Vec::with_capacity(n_left + 1);
    ptrs.push(0);
    let (mut indices, mut cells) = (Vec::new(), Vec::new());
    for i in 0..n_left {
        let row = &table[i * n_right..][..n_right];
        let sums = row.iter().enumerate().map(|(j, cell)| {
            let x = match C::CAN_CANCEL && cell.needs_exact(slack) {
                true => exact
                    .get_or_insert_with(|| exact_from(v, &table, i * n_right + j, slack, at))
                    .next()
                    .expect("an exact sum per cell needing one"),
                false => cell.sum(),
            };
            (j, x)
        });
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

/// The exact sums of the table's cells from `first` on that need one, from one more pass over `V`.
#[cold]
#[inline(never)]
fn exact_from<T: Real, J: PrimInt, I: Index, C: CellSum>(
    v: FactorWidth2<'_, T, J>,
    table: &[C],
    first: usize,
    slack: f64,
    at: impl Fn(&Column<T, I>) -> usize,
) -> std::vec::IntoIter<f64> {
    let wanted: Vec<usize> = (first..table.len())
        .filter(|&k| table[k].needs_exact(slack))
        .collect();
    let mut found = Vec::new();
    v.for_each_column::<I>(|column| {
        let k = at(&column);
        if wanted.binary_search(&k).is_ok() {
            found.push((k, column.cell()));
        }
    });
    exact_sums(found)
}

/// The exact sums of a row's `touched` cells that need one, from its bucket of columns and terms.
#[cold]
#[inline(never)]
fn exact_in_row<I: Index, C: CellSum>(
    touched: &[I],
    sums: &[C],
    slack: f64,
    (columns, cells): (&[I], &[f64]),
) -> std::vec::IntoIter<f64> {
    let wanted: Vec<I> = (touched.iter().copied())
        .filter(|j| sums[j.index()].needs_exact(slack))
        .collect();
    let found = (columns.iter().zip(cells))
        .filter(|&(j, _)| wanted.binary_search(j).is_ok())
        .map(|(j, &x)| (j.index(), x));
    exact_sums(found.collect())
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
    v.for_each_column::<I>(
        #[inline(always)]
        |column| {
            let (lo, hi) = (column.a.min(column.b), column.a.max(column.b));
            let slot = &mut cursor[lo.index()];
            columns[*slot] = hi - I::from_index(first_column);
            cells[*slot] = column.cell();
            *slot += 1;
        },
    );
    drop(cursor);
    let n_cols = v.n() - first_column;
    let slack = v.n_columns() as f64 * f64::EPSILON;
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
        // Summed before the writes below overwrite the bucket.
        let mut exact = (C::CAN_CANCEL
            && touched.iter().any(|j| sums[j.index()].needs_exact(slack)))
        .then(|| {
            let bucket = (&columns[span[0]..span[1]], &cells[span[0]..span[1]]);
            exact_in_row(&touched, &sums, slack, bucket)
        });
        // Written in place: a row's sums never outnumber its own bucket, already read.
        for &j in &touched {
            let sum = std::mem::take(&mut sums[j.index()]);
            let cell = match exact.as_mut() {
                Some(exact) if sum.needs_exact(slack) => {
                    exact.next().expect("an exact sum per cell needing one")
                }
                _ => sum.sum(),
            };
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
    drop((starts, sums));
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
        // A valued table's cap is half as many cells, so past it the smaller buckets win.
        let valued =
            |n_right| Accumulator::of::<f64, u32, Bounded>(2000, n_right, filling(2000 * n_right));
        assert_eq!(valued(DENSE_TABLE_MAX_CELLS / 4000), Accumulator::Table);
        assert_eq!(
            valued(DENSE_TABLE_MAX_CELLS / 4000 + 1),
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

    /// Cell (0, 0) sums to 1; plainly summed it is `−2⁵⁶` in the first order, and TwoSum's sum is 0.
    #[rstest::rstest]
    #[case::one_amid([p(110), p(56), -p(110), p(53), 1.0, -p(53), -p(56)])]
    #[case::one_last([p(110), -p(110), p(56), -p(56), p(53), -p(53), 1.0])]
    fn table_and_buckets_sum_a_cancelling_cell_exactly(#[case] terms: [f64; 7]) {
        let x = [&terms[..], &[1.0]].concat();
        let rows = [0u32, 0, 0, 0, 0, 0, 0, 1];
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &rows, Some(&x[..])).unwrap(),
            FactorWidth1::new(2, &rows, None).unwrap(),
        )
        .unwrap();
        for accumulator in [Accumulator::Table, Accumulator::Buckets] {
            let mut pairs = Vec::new();
            accumulator
                .cross::<f64, u32, u32, Bounded>(
                    gram.0,
                    2,
                    |_| {},
                    |i, j, cell| pairs.push((i, j, cell)),
                )
                .unwrap();
            assert_eq!(pairs, [(0, 2, 1.0), (1, 3, 1.0)], "{accumulator:?}");
        }
    }

    /// Its row holds 3 terms and `V` 3003: both accumulators judge the cell by `V`'s count.
    #[test]
    fn table_and_buckets_take_one_bound_for_a_cell_near_it() {
        let x = [&[1.0, 3.0 * p(-54), -1.0 + p(-40)][..], &[1.0; 3000]].concat();
        let left = [vec![0u32; 3], vec![1; 3000]].concat();
        let right = [vec![0u32; 3], vec![1; 3000]].concat();
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &left, Some(&x[..])).unwrap(),
            FactorWidth1::new(2, &right, None).unwrap(),
        )
        .unwrap();
        let sum = |accumulator: Accumulator| {
            let mut pairs = Vec::new();
            accumulator
                .cross::<f64, u32, u32, Bounded>(
                    gram.0,
                    2,
                    |_| {},
                    |i, j, cell| pairs.push((i, j, cell)),
                )
                .unwrap();
            pairs
        };
        assert_eq!(sum(Accumulator::Table), sum(Accumulator::Buckets));
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
