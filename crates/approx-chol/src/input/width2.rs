//! Factor-width-2 views: `A = V W Vᵀ + diag(g)`, column `k` of `V` joining `a[k]` and `b[k]`.

pub(super) mod bipartite;
pub(super) mod error;
pub(super) mod laplacian;
pub(super) mod weights;

use num_traits::PrimInt;

use self::error::{check_ends, length, ViewError};
use self::weights::Weighting;
use super::index;
use crate::{widen, Index, Real};

/// `V`'s columns and vertices, one variant per view.
#[derive(Clone, Copy, Debug)]
enum Columns<'a, T> {
    /// `e_a − e_b`, `a[k] ≠ b[k] < n`, and a [`Laplacian`](laplacian::Laplacian)'s ground.
    Incidence { n: usize, surplus: Option<&'a [T]> },
    /// `x[k] e_a + y[k] e_b`, `a[k] ≠ b[k] < n`.
    FactorWidth2 { n: usize, x: &'a [T], y: &'a [T] },
    /// `x[k] e_a + y[k] e_(n_left + b)`, a missing side `1`; with neither, read switched as `e_a − e_b`.
    Bipartite {
        n_left: usize,
        n_right: usize,
        x: Option<&'a [T]>,
        y: Option<&'a [T]>,
    },
}

/// `V` with at most two nonzeros per column: `x[k]` at `a[k]`, `y[k]` at `b[k] ≠ a[k]`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32> {
    a: &'a [I],
    b: &'a [I],
    weights: Weighting<&'a [T]>,
    columns: Columns<'a, T>,
}

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `n` vertices; lengths and `a[k] ≠ b[k] < n` are checked, so a zero value leaves one nonzero.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        x: &'a [T],
        y: &'a [T],
    ) -> Result<Self, ViewError> {
        length(a.len(), x.len())?;
        length(a.len(), y.len())?;
        check_ends(n, a, b)?;
        let columns = Columns::FactorWidth2 { n, x, y };
        Ok(Self {
            a,
            b,
            weights: Weighting::Unit,
            columns,
        })
    }

    /// The left part's size, after which `b` is numbered, when `V` is bipartite.
    pub(crate) fn n_left(&self) -> Option<usize> {
        match self.columns {
            Columns::Bipartite { n_left, .. } => Some(n_left),
            Columns::Incidence { .. } | Columns::FactorWidth2 { .. } => None,
        }
    }

    /// The ground `g`, which only a grounded [`Laplacian`](laplacian::Laplacian) has.
    pub(crate) fn surplus(&self) -> Option<&'a [T]> {
        match self.columns {
            Columns::Incidence { surplus, .. } => surplus,
            Columns::FactorWidth2 { .. } | Columns::Bipartite { .. } => None,
        }
    }

    /// `V` is an incidence matrix, so `A` is dominant by shape and its diagonal derived.
    pub(crate) fn is_incidence(&self) -> bool {
        matches!(
            self.columns,
            Columns::Incidence { .. }
                | Columns::Bipartite {
                    x: None,
                    y: None,
                    ..
                }
        )
    }

    /// Dimension of `A`, not the column count; saturates, as a sum past `usize` fails allocation anyway.
    pub(crate) fn n(&self) -> usize {
        match self.columns {
            Columns::Incidence { n, .. } | Columns::FactorWidth2 { n, .. } => n,
            Columns::Bipartite {
                n_left, n_right, ..
            } => n_left.saturating_add(n_right),
        }
    }

    /// Each column in order; write `visit` as `#[inline(always)] |column| …` (outlined: 1.6-1.9x).
    #[inline(always)]
    pub(crate) fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let offset = self.n_left().unwrap_or(0);
        let (one, minus_one) = (|_| T::one(), |_| -T::one());
        // Each kind loops straight: reading `V` through a per-column match cost within's paths 6%.
        match self.columns {
            Columns::Incidence { .. }
            | Columns::Bipartite {
                x: None, y: None, ..
            } => self.columns_of(offset, one, minus_one, visit),
            Columns::Bipartite {
                x: Some(x),
                y: None,
                ..
            } => self.columns_of(offset, |k| x[k], one, visit),
            Columns::Bipartite {
                x: None,
                y: Some(y),
                ..
            } => self.columns_of(offset, one, |k| y[k], visit),
            Columns::FactorWidth2 { x, y, .. }
            | Columns::Bipartite {
                x: Some(x),
                y: Some(y),
                ..
            } => self.columns_of(offset, |k| x[k], |k| y[k], visit),
        }
    }

    /// Column `k` is `v_a(k) e_a + v_b(k) e_(offset + b)`; matched once, then looped per case.
    #[inline(always)]
    fn columns_of<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        visit: impl FnMut(Column<T, S>),
    ) {
        match self.weights {
            Weighting::Unit => self.weighted(offset, v_a, v_b, |_| Weighting::Unit, visit),
            Weighting::Of(w) => self.weighted(offset, v_a, v_b, |k| Weighting::Of(w[k]), visit),
            Weighting::SqrtOf(s) => {
                self.weighted(offset, v_a, v_b, |k| Weighting::SqrtOf(s[k]), visit)
            }
        }
    }

    #[inline(always)]
    fn weighted<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weighting<T>,
        mut visit: impl FnMut(Column<T, S>),
    ) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: S::from_index(index(*a)),
                b: S::from_index(offset + index(*b)),
                v_a: v_a(k),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}

/// Column `k` of `V` under its weight, unfolded, so the floating test reads `V` exactly as given.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Column<T, I> {
    pub(crate) a: I,
    pub(crate) b: I,
    pub(crate) v_a: T,
    pub(crate) v_b: T,
    pub(crate) weight: Weighting<T>,
}

impl<T: Real, I> Column<T, I> {
    /// `w·v_a·v_b` in `f64`: every assembly sums its cells there and casts once.
    pub(crate) fn cell(self) -> f64 {
        (self.weight).bilinear_unflushed(widen(self.v_a), widen(self.v_b))
    }
}
