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
    /// Checks the arrays' shape only.
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
    /// A column index is at least `n`, or a row's columns are not ascending.
    MalformedRow {
        /// The offending row.
        row: usize,
    },
    /// An entry has no equal mirror.
    NotSymmetric {
        /// The offending row.
        row: usize,
    },
    /// An entry is not finite.
    NotFinite {
        /// The offending row.
        row: usize,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid CSR: {self:?}")
    }
}

impl std::error::Error for CsrError {}
