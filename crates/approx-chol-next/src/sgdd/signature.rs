use super::{Congruence, OffDiagonal};
use crate::storage::CrossBlock;
use crate::Real;

/// The signs `S` folding `A_c` into a Z-matrix; `None` when frustrated.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    match off_diagonal {
        OffDiagonal::Bipartite(cross) if cross.positive() => {
            Some(Congruence::negate_from(cross.n_eliminated()))
        }
        OffDiagonal::Bipartite(cross) => folding_signs(cross),
        OffDiagonal::General(adjacency) => {
            let mut forest = ParityForest::new(adjacency.n());
            for i in 0..adjacency.n() {
                for (j, &a) in adjacency.upper(i) {
                    forest.join(i, j, a > T::zero())?;
                }
            }
            Some(forest.signs())
        }
    }
}

/// `None` when the component is frustrated: a negative cycle admits no signature; parities by union-find over `C` alone, as a search through `Cᵀ` cost a transpose `scaled` drops.
fn folding_signs<T: Real>(cross: &CrossBlock<T>) -> Option<Congruence<T>> {
    let n_eliminated = cross.n_eliminated();
    let mut forest = ParityForest::new(cross.n());
    for e in 0..n_eliminated {
        for (k, &c) in cross.c().row(e) {
            // A positive entry folds only across opposite signs.
            forest.join(e, n_eliminated + k, c > T::zero())?;
        }
    }
    Some(forest.signs())
}

/// Union-find whose links carry whether the two ends' signs differ.
struct ParityForest {
    parent: Vec<u32>,
    /// Parity to the parent; `false` at a root.
    odd: Vec<bool>,
    size: Vec<u32>,
}

impl ParityForest {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            odd: vec![false; n],
            size: vec![1; n],
        }
    }

    /// Root and parity to it, halving the path.
    fn find(&mut self, mut x: usize) -> (usize, bool) {
        let mut parity = false;
        while self.parent[x] as usize != x {
            let up = self.parent[x] as usize;
            self.odd[x] ^= self.odd[up];
            self.parent[x] = self.parent[up];
            parity ^= self.odd[x];
            x = self.parent[x] as usize;
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
        self.parent[small] = big as u32;
        self.odd[small] = pi ^ pj ^ differ;
        self.size[big] += self.size[small];
        Some(())
    }
}
