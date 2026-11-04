//! A matrix's grounding and balance: the surplus closing it into an SDDM, and how many sheets it needs.

use super::{in_range, widen, Real};

/// Whether a signature folds `M` into a Z-matrix (Harary): balanced `M` is singular exactly when ungrounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Balance {
    Balanced,
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even ungrounded.
    Frustrated,
}

impl Balance {
    /// A balanced `M` folds onto one sheet; a frustrated one needs the cover's two.
    pub(crate) fn sheets(self) -> Sheets {
        match self {
            Self::Balanced => Sheets::One,
            Self::Frustrated => Sheets::Two,
        }
    }
}

/// A matrix's off-diagonal and the surplus closing it into an SDDM.
#[derive(Clone, Debug)]
pub(crate) struct Grounded<T, P> {
    pub(crate) adjacency: P,
    pub(crate) grounding: Grounding<T>,
}

/// Where a floating solve fixes its gauge, on the vertices Cholesky factors: the largest scale, and whether the scales span past `1/√ε`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gauge {
    pub(crate) peak: usize,
    pub(crate) spans_wide: bool,
}

/// How many copies of each vertex a matrix holds: two for a Gremban cover, interleaved by vertex.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sheets {
    One,
    Two,
}

impl Sheets {
    pub(crate) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// The surplus `s = diag(M) − Σ|offdiag(M)|` closing `M` into an SDDM; none is `σ = 0`, singular only when balanced.
#[derive(Clone, Debug)]
pub(crate) struct Grounding<T>(Option<Box<[T]>>);

impl<T: Real> Grounding<T> {
    /// Panics on a negative or non-finite entry; ungrounded exactly when every entry is zero.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        let valid = |&s: &T| s.is_finite() && s >= T::zero();
        assert!(
            surplus.iter().all(valid),
            "a surplus is finite and nonnegative"
        );
        match surplus.iter().all(|&s| s == T::zero()) {
            true => Self(None),
            false => Self(Some(surplus.into())),
        }
    }

    /// Each entry below the range floor (#163) set to zero while within `band · d_i`; past it, `Err` names the row of a real ground out of range.
    pub(crate) fn flushed(self, diagonal: &[T], band: f64) -> Result<Self, usize> {
        let Some(mut surplus) = self.0 else {
            return Ok(self);
        };
        for (i, (s, &d)) in surplus.iter_mut().zip(diagonal).enumerate() {
            if !in_range(*s) {
                if widen(*s) > band * widen(d) {
                    return Err(i);
                }
                *s = T::zero();
            }
        }
        Ok(Self::of(surplus.into_vec()))
    }

    pub(crate) fn ungrounded() -> Self {
        Self(None)
    }

    pub(crate) fn surplus(&self) -> Option<&[T]> {
        self.0.as_deref()
    }

    /// Row `i`'s ground edge; a zero surplus couples nothing, so it is none.
    pub(crate) fn edge(&self, i: usize) -> Option<T> {
        self.surplus().map(|s| s[i]).filter(|&s| s > T::zero())
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.0.is_some()
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.0.as_ref().is_none_or(|surplus| surplus.len() == n)
    }
}
