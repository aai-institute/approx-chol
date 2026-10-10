use num_traits::PrimInt;

/// Borrowed CSR claiming a symmetric `A`; the claim is checked when it becomes an [`Sgdd`](crate::Sgdd).
#[derive(Clone, Copy, Debug)]
pub struct Csr<'a, T, I = u32> {
    n: usize,
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
}

impl<'a, T, I: PrimInt> Csr<'a, T, I> {
    /// Checks the arrays' shape, that `row_ptrs` ascends from `0`, and that every vertex and entry fits a `u32` index, so no row slice can fail.
    pub fn new(
        n: usize,
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
    ) -> Result<Self, CsrError> {
        todo!()
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Visits each nonzero `(i, j)`, `i < j`, checking mirrors by a monotone per-row cursor (search: +6-15% build).
    pub(crate) fn walk(&self, edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError>
    where
        T: crate::Real,
    {
        todo!()
    }

    /// Row `i` as the caller stored it, unchecked beyond [`new`](Self::new)'s shape.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, &'a T)> + '_ {
        let index = |k: I| k.to_usize().expect("a shape-checked index fits usize");
        let span = index(self.row_ptrs[i])..index(self.row_ptrs[i + 1]);
        (self.col_indices[span.clone()]
            .iter()
            .map(move |&j| index(j)))
        .zip(&self.values[span])
    }
}

/// Why a CSR is not a symmetric matrix; `row` is in the caller's numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrError {
    /// An array's length disagrees with `n` or with `row_ptrs`.
    LengthMismatch {
        /// The length the other arrays fix.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// More vertices or entries than a `u32` index reaches.
    TooLarge {
        /// The count that overflows.
        count: usize,
    },
    /// `row_ptrs` does not start at `0`, or falls or turns negative at `row`.
    RowPointers {
        /// The first row whose span is not a range.
        row: u32,
    },
    /// A column index lies outside `0..n`.
    ColumnOutOfRange {
        /// The offending row.
        row: u32,
    },
    /// A row's columns do not strictly ascend: rows must be canonical, a repeated column is not summed.
    NotAscending {
        /// The offending row.
        row: u32,
    },
    /// An entry has no equal mirror.
    NotSymmetric {
        /// The offending row.
        row: u32,
    },
    /// An entry is not finite, or nonzero outside `[MIN_POSITIVE/EPSILON, MAX·EPSILON]`, where products turn subnormal or row sums overflow.
    OutOfRange {
        /// The offending row.
        row: u32,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "CSR array has length {found}, expected {expected}")
            }
            Self::TooLarge { count } => write!(f, "{count} exceeds a u32 index"),
            Self::RowPointers { row } => {
                write!(f, "row pointers do not ascend from 0 at row {row}")
            }
            Self::ColumnOutOfRange { row } => {
                write!(f, "row {row} has a column index outside 0..n")
            }
            Self::NotAscending { row } => write!(
                f,
                "row {row} has unsorted or repeated columns; scipy's sum_duplicates() canonicalizes"
            ),
            Self::NotSymmetric { row } => write!(f, "row {row} has an entry with no equal mirror"),
            Self::OutOfRange { row } => {
                write!(f, "row {row} has an entry outside the representable range")
            }
        }
    }
}

impl std::error::Error for CsrError {}
