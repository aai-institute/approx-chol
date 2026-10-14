use super::congruence::Congruence;
use crate::storage::{OffDiagonal, Rows};
use crate::{Index, Real};

/// The signs `S` folding `A_c` into a Z-matrix, `I` when it already is one; `None` when frustrated.
pub(super) fn folding_signs<T: Real, I: Index>(
    off_diagonal: &OffDiagonal<T, I>,
) -> Option<Congruence<T>> {
    if off_diagonal.nonpositive() {
        return Some(Congruence::Identity);
    }
    match off_diagonal {
        OffDiagonal::Bipartite(cross) if cross.positive() => {
            Some(Congruence::NegateFrom(cross.n_rows()))
        }
        OffDiagonal::Bipartite(cross) => cross_signs(cross),
        OffDiagonal::General(adjacency) => {
            let mut forest = ParityForest::<I>::new(adjacency.n());
            for i in 0..adjacency.n() {
                for (j, &a) in adjacency.upper(i) {
                    forest.join(i, j, a > T::zero())?;
                }
            }
            Some(forest.signs())
        }
    }
}

/// `None` when frustrated; union-find over `C` alone, as searching `Cᵀ` costs a transpose.
fn cross_signs<T: Real, I: Index>(cross: &Rows<T, I>) -> Option<Congruence<T>> {
    let n_eliminated = cross.n_rows();
    let mut forest = ParityForest::<I>::new(n_eliminated + cross.n_cols());
    for e in 0..n_eliminated {
        for (k, &c) in cross.row(e) {
            // A positive entry folds only across opposite signs.
            forest.join(e, n_eliminated + k, c > T::zero())?;
        }
    }
    Some(forest.signs())
}

/// Union-find whose links carry whether the two ends' signs differ.
struct ParityForest<I> {
    parent: Vec<I>,
    /// Parity to the parent; `false` at a root.
    odd: Vec<bool>,
    size: Vec<I>,
}

impl<I: Index> ParityForest<I> {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).map(I::from_index).collect(),
            odd: vec![false; n],
            size: vec![I::one(); n],
        }
    }

    /// Root and parity to it, halving the path.
    fn find(&mut self, mut x: usize) -> (usize, bool) {
        let mut parity = false;
        while self.parent[x].index() != x {
            let up = self.parent[x].index();
            self.odd[x] ^= self.odd[up];
            self.parent[x] = self.parent[up];
            parity ^= self.odd[x];
            x = self.parent[x].index();
        }
        (x, parity)
    }

    /// Each vertex negated when odd to its root.
    fn signs<T: Real>(mut self) -> Congruence<T> {
        let negated: Vec<bool> = (0..self.parent.len()).map(|i| self.find(i).1).collect();
        Congruence::negated(&negated)
    }

    /// `None` when `i` and `j` are already joined at the other parity.
    fn join(&mut self, i: usize, j: usize, differ: bool) -> Option<()> {
        let ((ri, pi), (rj, pj)) = (self.find(i), self.find(j));
        if ri == rj {
            return (pi ^ pj == differ).then_some(());
        }
        let (big, small) = match self.size[ri] >= self.size[rj] {
            true => (ri, rj),
            false => (rj, ri),
        };
        self.parent[small] = I::from_index(big);
        self.odd[small] = pi ^ pj ^ differ;
        self.size[big] = self.size[big] + self.size[small];
        Some(())
    }
}
