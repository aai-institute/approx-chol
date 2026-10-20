//! `A = V W Vᵀ + diag(σ)` as every factor-width-2 view lowers it: column `k` is `tail(k) e_a + head(k) e_b`.

use num_traits::PrimInt;

use super::edges::Edges;
use super::{Column, IntoColumns};
use crate::{Index, Real};

/// What every entry of `V` at one end is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entries {
    /// `+1` in every column.
    Plus,
    /// `−1` in every column.
    Minus,
    /// `±1`, per column.
    Signs,
    /// Any value, per column.
    Values,
}

/// One end's entries of `V`, read per column.
pub(crate) trait End<T>: Copy {
    const ENTRIES: Entries;

    fn at(self, k: usize) -> T;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct One;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MinusOne;

/// `+1` where `positive[k]`, else `−1`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Incidences<'a>(pub(super) &'a [bool]);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Values<'a, T>(pub(super) &'a [T]);

/// Another end's entries, negated.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Negated<E>(E);

impl<T: Real> End<T> for One {
    const ENTRIES: Entries = Entries::Plus;

    #[inline(always)]
    fn at(self, _: usize) -> T {
        T::one()
    }
}

impl<T: Real> End<T> for MinusOne {
    const ENTRIES: Entries = Entries::Minus;

    #[inline(always)]
    fn at(self, _: usize) -> T {
        -T::one()
    }
}

impl<T: Real> End<T> for Incidences<'_> {
    const ENTRIES: Entries = Entries::Signs;

    #[inline(always)]
    fn at(self, k: usize) -> T {
        match self.0[k] {
            true => T::one(),
            false => -T::one(),
        }
    }
}

impl<T: Real> End<T> for Values<'_, T> {
    const ENTRIES: Entries = Entries::Values;

    #[inline(always)]
    fn at(self, k: usize) -> T {
        self.0[k]
    }
}

impl<T: Real, E: End<T>> End<T> for Negated<E> {
    const ENTRIES: Entries = match E::ENTRIES {
        Entries::Plus => Entries::Minus,
        Entries::Minus => Entries::Plus,
        entries => entries,
    };

    #[inline(always)]
    fn at(self, k: usize) -> T {
        -self.0.at(k)
    }
}

/// How `V`'s two ends index `A`.
pub(crate) trait Graph: Copy {
    const BIPARTITE: bool;

    /// Where the head's part starts in `A`'s numbering.
    fn head_offset(self) -> usize;
}

/// Both ends anywhere in `0..n`, never the same row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct General;

/// The tail in the left part, the head in the right one, numbered after it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bipartite {
    pub(crate) n_left: usize,
}

impl Graph for General {
    const BIPARTITE: bool = false;

    fn head_offset(self) -> usize {
        0
    }
}

impl Graph for Bipartite {
    const BIPARTITE: bool = true;

    fn head_offset(self) -> usize {
        self.n_left
    }
}

/// The signs folding `A` into a Z-matrix when `V`'s shape fixes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fold {
    /// Every cell is already nonpositive.
    Identity,
    /// Every cell is positive and crosses the parts: reading the head negated folds it.
    NegatedHead,
}

/// `A = V W Vᵀ + diag(σ)` on `n` vertices, `σ` none when every entry is zero.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gram<'a, T, I, G, A, B> {
    n: usize,
    graph: G,
    edges: Edges<'a, T, I>,
    tail: A,
    head: B,
    diagonal: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt, G: Graph, A: End<T>, B: End<T>> Gram<'a, T, I, G, A, B> {
    /// The view checked every length and index.
    pub(super) fn new(
        n: usize,
        graph: G,
        edges: Edges<'a, T, I>,
        tail: A,
        head: B,
        diagonal: Option<&'a [T]>,
    ) -> Self {
        Self {
            n,
            graph,
            edges,
            tail,
            head,
            diagonal,
        }
    }

    pub(crate) fn graph(&self) -> G {
        self.graph
    }

    /// `σ`, which is no column of `V`.
    pub(crate) fn diagonal(&self) -> Option<&'a [T]> {
        self.diagonal
    }

    /// D: every entry of `V` is `±1`, so `A` is dominant by shape and its diagonal derived.
    pub(crate) fn is_incidence(&self) -> bool {
        A::ENTRIES != Entries::Values && B::ENTRIES != Entries::Values
    }

    /// X: two parallel columns can put opposite signs on one cell.
    pub(crate) fn cancels(&self) -> bool {
        self.is_incidence() && (A::ENTRIES == Entries::Signs || B::ENTRIES == Entries::Signs)
    }

    /// K: the fold, when every cell's sign is fixed by the ends and the graph.
    pub(crate) fn fold(&self) -> Option<Fold> {
        use Entries::{Minus, Plus};
        match (A::ENTRIES, B::ENTRIES) {
            (Plus, Minus) | (Minus, Plus) => Some(Fold::Identity),
            (Plus, Plus) | (Minus, Minus) if G::BIPARTITE => Some(Fold::NegatedHead),
            _ => None,
        }
    }

    /// The same columns with the head negated: `S A S` for `S` negating the head's part.
    pub(crate) fn negated_head(self) -> Gram<'a, T, I, G, A, Negated<B>> {
        Gram {
            n: self.n,
            graph: self.graph,
            edges: self.edges,
            tail: self.tail,
            head: Negated(self.head),
            diagonal: self.diagonal,
        }
    }
}

impl<T: Real, I: PrimInt, G: Graph, A: End<T>, B: End<T>> IntoColumns<T>
    for Gram<'_, T, I, G, A, B>
{
    fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let (tail, head) = (self.tail, self.head);
        (self.edges).for_each_column(
            self.graph.head_offset(),
            |k| tail.at(k),
            |k| head.at(k),
            visit,
        )
    }
}

/// What runs on a bipartite Gram, whichever ends its sides give `V`.
pub(crate) trait OnBipartite<'a, T, I> {
    type Output;

    fn run<A: End<T>, B: End<T>>(self, gram: Gram<'a, T, I, Bipartite, A, B>) -> Self::Output;
}
