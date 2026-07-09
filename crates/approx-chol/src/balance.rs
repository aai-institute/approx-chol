//! Fold a signed matrix's off-diagonal signs into the Laplacian convention
//! (`sᵢ·Mᵢⱼ·sⱼ ≤ 0`), or witness a frustrated cycle.

use crate::graph::{classify, Entry};
use crate::{CsrRef, Error};
use num_traits::Float;

/// Per-node ±1 signs folding every off-diagonal into the Laplacian convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature(Vec<i8>);

impl Signature {
    /// The per-node signs, each `+1` or `-1`.
    #[must_use]
    pub fn signs(&self) -> &[i8] {
        &self.0
    }
}

/// Certify that `csr`'s off-diagonal signs are balanceable, returning a folding
/// [`Signature`] (trivial for sign-free input).
///
/// # Errors
///
/// [`Error::FrustratedSigns`] when no signature folds every edge; CSR
/// validation errors otherwise.
pub fn certify_balance<T>(csr: CsrRef<'_, T, u32>) -> Result<Signature, Error>
where
    T: Float + Send + Sync + 'static,
{
    csr.validate()?;
    let n = csr.n();
    let mut dsu = ParityDsu::new(n);
    let mut any_positive = false;
    for row in 0..n {
        let (cols, vals) = csr.try_row(row)?;
        for (&col, &val) in cols.iter().zip(vals.iter()) {
            let col = col as usize;
            if row >= col {
                continue;
            }
            let opposite = match classify(row, col, val) {
                Entry::Edge => false,
                Entry::PositiveOffDiagonal => true,
                Entry::Diagonal | Entry::StructuralZero => continue,
            };
            any_positive |= opposite;
            if !dsu.constrain(row, col, opposite) {
                return Err(Error::FrustratedSigns { edge: (row, col) });
            }
        }
    }
    if !any_positive {
        return Ok(Signature(vec![1; n]));
    }
    Ok(dsu.signature())
}

struct ParityDsu {
    parent: Vec<u32>,
    rank: Vec<u32>,
    /// Sign parity to parent: `true` = opposite sign, `false` = same.
    rel: Vec<bool>,
}

impl ParityDsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
            rel: vec![false; n],
        }
    }

    /// Union by rank bounds tree height at `log₂(n)`, so this recursion is stack-safe.
    fn find(&mut self, x: usize) -> (usize, bool) {
        let p = self.parent[x] as usize;
        if p == x {
            return (x, false);
        }
        let (root, parent_parity) = self.find(p);
        let parity = self.rel[x] ^ parent_parity;
        self.parent[x] = root as u32;
        self.rel[x] = parity;
        (root, parity)
    }

    /// Returns `false` if the constraint conflicts with an existing one.
    fn constrain(&mut self, a: usize, b: usize, opposite: bool) -> bool {
        let (ra, pa) = self.find(a);
        let (rb, pb) = self.find(b);
        if ra == rb {
            return (pa ^ pb) == opposite;
        }
        let rel = pa ^ pb ^ opposite;
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb as u32;
            self.rel[ra] = rel;
        } else {
            self.parent[rb] = ra as u32;
            self.rel[rb] = rel;
            if self.rank[ra] == self.rank[rb] {
                self.rank[ra] += 1;
            }
        }
        true
    }

    fn signature(mut self) -> Signature {
        let n = self.parent.len();
        Signature(
            (0..n)
                .map(|i| if self.find(i).1 { -1 } else { 1 })
                .collect(),
        )
    }
}
