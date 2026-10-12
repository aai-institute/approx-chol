//! `A` whole, as each class it passes through on the way to [`Sgdd`]: an input enters at the class its type proves.

use std::cell::LazyCell;

use super::decomposition::{ground_or_float, Partition};
use super::null_vectors::{NullVectors, OnDemand};
use super::{
    within_series_floor, Congruence, Congruent, Dominant, Folded, ReduceError, RowSums,
    ScalingPolicy, Sgdd, Unreduced,
};
use crate::input::IntoColumns;
use crate::Real;

/// `⊕ A_c`, each with its diagonal as the input gave it: dominance still to be shown.
pub(super) struct Gdd<T> {
    pub(super) partition: Partition,
    pub(super) blocks: Vec<Unreduced<T>>,
    pub(super) dead: Vec<u32>,
}

/// `⊕ A_c`, each with the scales `R` making it dominant once clamped, not yet applied: its signature is still open.
pub(super) struct Scaled<T> {
    n: usize,
    blocks: Vec<Dominant<T>>,
    dead: Vec<u32>,
}

/// `⊕ M_c`, each formed once with its rows and split by balance: only its surplus is left.
pub(super) struct Signed<T> {
    n: usize,
    folded: Vec<(Folded<T>, RowSums<T>)>,
    frustrated: Vec<(Congruent<T>, RowSums<T>)>,
    dead: Vec<u32>,
}

/// Each component's block, `None` when it is dead; a live one past `1/ε` vertices of `T` is rejected.
pub(super) fn extract<T: Real, B>(
    partition: &Partition,
    mut block: impl FnMut(&[u32]) -> Result<Option<B>, ReduceError>,
) -> Result<(Vec<B>, Vec<u32>), ReduceError> {
    let (mut blocks, mut dead) = (Vec::new(), Vec::new());
    for vertices in partition.components() {
        match block(vertices)? {
            Some(b) => {
                within_series_floor::<T>(vertices)?;
                blocks.push(b);
            }
            None => dead.extend_from_slice(vertices),
        }
    }
    Ok((blocks, dead))
}

impl<T: Real> Gdd<T> {
    /// Rows prove nothing: each is scaled past its row test, its existence unknown.
    pub(super) fn scale(self, policy: &ScalingPolicy) -> Result<Scaled<T>, ReduceError> {
        let blocks = (self.blocks.into_iter())
            .map(|a| a.scale(policy))
            .collect::<Result<_, _>>()?;
        Ok(Scaled {
            n: self.partition.n(),
            blocks,
            dead: self.dead,
        })
    }

    /// `V` proves each scalable; its null vectors are built by the first component that is not already dominant.
    pub(super) fn scale_gram(
        self,
        policy: &ScalingPolicy,
        v: impl IntoColumns<T>,
    ) -> Result<Scaled<T>, ReduceError> {
        let Self {
            partition,
            blocks,
            dead,
        } = self;
        let nulls: OnDemand<'_> =
            LazyCell::new(Box::new(|| NullVectors::of(v, partition.components())));
        let blocks = (blocks.into_iter())
            .map(|a| {
                let scaling = a.gram_scaling(policy, &nulls)?;
                Ok(Dominant { a, scaling })
            })
            .collect::<Result<_, _>>()?;
        drop(nulls);
        Ok(Scaled {
            n: partition.n(),
            blocks,
            dead,
        })
    }
}

impl<T: Real> Scaled<T> {
    /// Each signature searched, then `M` formed with `F = S R`, folded or frustrated.
    pub(super) fn sign(self) -> Result<Signed<T>, ReduceError> {
        let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
        for a in self.blocks {
            match a.signature() {
                Some(signs) => {
                    let (m, rows) = a.congruent(signs)?;
                    folded.push((Folded(m), rows));
                }
                None => frustrated.push(a.congruent(Congruence::identity())?),
            }
        }
        Ok(Signed {
            n: self.n,
            folded,
            frustrated,
            dead: self.dead,
        })
    }
}

impl<T: Real> Signed<T> {
    /// Rows prove nothing more: every `M` grounds on its own rows.
    pub(super) fn by_rows(self) -> Result<Sgdd<T>, ReduceError> {
        let folded = (self.folded.into_iter()).map(|(m, rows)| m.balanced(rows.surplus()));
        let frustrated =
            (self.frustrated.into_iter()).map(|(m, rows)| m.frustrated(rows.surplus()));
        Ok(Sgdd {
            n: self.n,
            components: folded.chain(frustrated).collect::<Result<_, _>>()?,
            dead: self.dead.into_boxed_slice(),
        })
    }

    /// `V` grounds every folded `M` on `M𝟙` in one pass; a frustrated one grounds on its rows.
    pub(super) fn by_columns(self, v: impl IntoColumns<T>) -> Result<Sgdd<T>, ReduceError> {
        let folded = self.folded.into_iter().map(|(m, _)| m).collect();
        let mut components = ground_or_float(v, folded)?;
        for (m, rows) in self.frustrated {
            components.push(m.frustrated(rows.surplus())?);
        }
        Ok(Sgdd {
            n: self.n,
            components,
            dead: self.dead.into_boxed_slice(),
        })
    }
}
