//! Partitioned matrices reduced component by component; folded Grams share one column pass.

use std::cell::LazyCell;

use super::decomposition::{ground_or_float, Partition};
use super::null_vectors::{LazyNullVectors, NullVectors};
use super::{scaling, signature, Existence};
use super::{
    within_series_floor, Balance, Component, Congruence, Frame, Grounding, OffDiagonal,
    ReduceError, ScalingPolicy, Sdd, Sgdd,
};
use crate::input::IntoColumns;
use crate::Real;

/// `⊕ A_c`, each with its input diagonal and connected vertices.
pub(super) struct DirectSum<T> {
    pub(super) partition: Partition,
    pub(super) blocks: Vec<SymmetricBlock<T>>,
    pub(super) dead: Vec<u32>,
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

impl<T: Real> DirectSum<T> {
    /// Scales and signs each component, closing its surplus from the same rows.
    pub(super) fn by_rows(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
        for a in self.blocks {
            match a.scale(policy)?.sign()? {
                Signed::Folded(m, rows) => folded.push(m.balanced(rows.surplus())?),
                Signed::Frustrated(component) => frustrated.push(component),
            }
        }
        folded.extend(frustrated);
        Ok(Sgdd {
            n: self.partition.n(),
            components: folded,
            dead: self.dead.into_boxed_slice(),
        })
    }

    /// Null vectors scale on demand; only folded components await the shared column surplus.
    pub(super) fn by_columns(
        self,
        policy: &ScalingPolicy,
        v: impl IntoColumns<T>,
    ) -> Result<Sgdd<T>, ReduceError> {
        let Self {
            partition,
            blocks,
            dead,
        } = self;
        let nulls: LazyNullVectors<'_> =
            LazyCell::new(Box::new(|| NullVectors::of(v, partition.components())));
        let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
        for a in blocks {
            match a.scale_gram(policy, &nulls)?.sign()? {
                Signed::Folded(m, _) => folded.push(m),
                Signed::Frustrated(component) => frustrated.push(component),
            }
        }
        let mut components = ground_or_float(v, folded)?;
        components.extend(frustrated);
        Ok(Sgdd {
            n: partition.n(),
            components,
            dead: dead.into_boxed_slice(),
        })
    }
}

/// One component `A_c` as the input gave it: signed, unscaled.
pub(super) struct SymmetricBlock<T> {
    /// In `A`'s numbering, in local order.
    pub(super) vertices: Box<[u32]>,
    pub(super) off_diagonal: OffDiagonal<T>,
    pub(super) diagonal: Vec<T>,
    /// `Σ_j |a_ij|`, summed in `f64` while its rows are extracted, so dominance and surplus reread no value and a long `f32` row is not undercounted.
    pub(super) abs_row_sums: Vec<T>,
    /// Entries per row: the rounding its sum can carry, read by the dominance test and the surplus band alike.
    pub(super) degrees: Vec<usize>,
}

impl<T: Real> SymmetricBlock<T> {
    /// Panics unless one vertex and one diagonal per row.
    pub(super) fn new(
        vertices: Box<[u32]>,
        off_diagonal: OffDiagonal<T>,
        diagonal: Vec<T>,
    ) -> Self {
        assert_eq!(vertices.len(), off_diagonal.n(), "one vertex per row");
        assert_eq!(diagonal.len(), off_diagonal.n(), "one diagonal per row");
        Self {
            vertices,
            abs_row_sums: off_diagonal.abs_row_sums(),
            degrees: off_diagonal.degrees(),
            off_diagonal,
            diagonal,
        }
    }

    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    pub(super) fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    /// Rows prove nothing: SCI past the row test, or the reduced CG on a bipartite block, its existence unknown.
    fn scale(self, policy: &ScalingPolicy) -> Result<Dominant<T>, ReduceError> {
        let certification = scaling::Certification::new(*policy, Existence::Unknown);
        let scaling =
            scaling::dominance_scaling(&self, certification, || match &self.off_diagonal {
                OffDiagonal::General(adjacency) => scaling::sci(adjacency, &self.diagonal, policy),
                OffDiagonal::Bipartite(cross) => scaling::reduced_cg(cross, &self.diagonal, policy),
            })?;
        Ok(Dominant { a: self, scaling })
    }

    /// `V` proves a scaling exists: a general block's certified null vector, else SCI; a bipartite block's reduced CG. A zero diagonal beside a cell is its terms underflowing, not a non-GDD input.
    fn scale_gram(
        self,
        policy: &ScalingPolicy,
        nulls: &LazyNullVectors<'_>,
    ) -> Result<Dominant<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| d == T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let certification = scaling::Certification::new(*policy, Existence::Guaranteed);
        let scaling =
            scaling::dominance_scaling(&self, certification, || match &self.off_diagonal {
                OffDiagonal::General(adjacency) => (LazyCell::force(nulls).on(&self.vertices))
                    .and_then(|u| scaling::null_scaling(&u, &self.diagonal))
                    .or_else(|| scaling::sci(adjacency, &self.diagonal, policy)),
                OffDiagonal::Bipartite(cross) => scaling::reduced_cg(cross, &self.diagonal, policy),
            })?;
        Ok(Dominant { a: self, scaling })
    }
}

/// `A_c` and the scales `R` making `R A_c R` weakly dominant once clamped, not yet applied: its signature is still open.
pub(super) struct Dominant<T> {
    a: SymmetricBlock<T>,
    scaling: scaling::Scaling<T>,
}

impl<T: Real> Dominant<T> {
    fn sign(self) -> Result<Signed<T>, ReduceError> {
        match folding_signs(&self.a.off_diagonal) {
            Some(signs) => {
                let (m, rows) = self.congruent(signs)?;
                Ok(Signed::Folded(Folded(m), rows))
            }
            None => {
                let (m, rows) = self.congruent(Congruence::identity())?;
                Ok(Signed::Frustrated(m.frustrated(rows.surplus())?))
            }
        }
    }

    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the report; a frustrated `A_c` takes `S = I`.
    fn congruent(self, signs: Congruence<T>) -> Result<(Congruent<T>, RowSums<T>), ReduceError> {
        let Self { a, scaling } = self;
        match scaling {
            scaling::Scaling::AlreadyDominant => {
                // Signs keep every magnitude, so `A_c`'s rows are `M`'s; its deficit lies inside the band `already_dominant` allowed.
                let rows = RowSums {
                    diagonal: a.diagonal,
                    abs: a.abs_row_sums,
                    surplus: RowSurplus::Banded(a.degrees),
                };
                Ok((Congruent::new(a.vertices, a.off_diagonal, signs)?, rows))
            }
            scaling::Scaling::Scaled(relaxation, certify) => {
                let vertex = a.vertices[0];
                let congruence = signs.scaled(&relaxation.scales);
                let (off_diagonal, abs) = (congruence.transform_abs_row_sums(a.off_diagonal))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal: Vec<T> = (a.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let deficit = scaling::clamped_deficit(&diagonal, &abs, &a.degrees);
                let m = Congruent {
                    frame: Frame::new(a.vertices, congruence),
                    off_diagonal,
                    report: certify.report(&relaxation, deficit),
                };
                let rows = RowSums {
                    diagonal,
                    abs,
                    surplus: RowSurplus::Residue,
                };
                Ok((m, rows))
            }
        }
    }
}

/// A signed component: only a folded matrix can still need a column-derived surplus.
enum Signed<T> {
    Folded(Folded<T>, RowSums<T>),
    Frustrated(Component<T>),
}

/// The signs `S` folding `A_c`; `None` when frustrated. A Z-matrix folds as it is.
pub(super) fn folding_signs<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    match off_diagonal.nonpositive() {
        true => Some(Congruence::identity()),
        false => signature::signature(off_diagonal),
    }
}

/// `M = F · A_c · F` over its frame, its surplus still to be decided.
pub(super) struct Congruent<T> {
    pub(super) frame: Frame<T>,
    pub(super) off_diagonal: OffDiagonal<T>,
    pub(super) report: Option<super::ScalingReport>,
}

impl<T: Real> Congruent<T> {
    /// `F A_c F` unscaled, so nothing is reported.
    pub(super) fn new(
        vertices: Box<[u32]>,
        off_diagonal: OffDiagonal<T>,
        congruence: Congruence<T>,
    ) -> Result<Self, ReduceError> {
        let vertex = vertices[0];
        let off_diagonal =
            (congruence.transform(off_diagonal)).ok_or(ReduceError::Unscalable { vertex })?;
        Ok(Self {
            frame: Frame::new(vertices, congruence),
            off_diagonal,
            report: None,
        })
    }

    pub(super) fn frustrated(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.grounded(Balance::Frustrated(grounding))
    }

    /// The ground's `Σs` must clear the bound any diagonal does.
    fn grounded(self, balance: Balance<T>) -> Result<Component<T>, ReduceError> {
        if !balance.ground_in_range() {
            let vertex = self.frame.vertices()[0];
            return Err(ReduceError::OutOfRange { vertex });
        }
        Ok(Component::new(
            self.frame,
            Sdd::new(self.off_diagonal, balance),
            self.report,
        ))
    }
}

/// `M` folded to a Z-matrix by a found signature or by its shape: only it grounds as balanced.
pub(super) struct Folded<T>(pub(super) Congruent<T>);

impl<T: Real> Folded<T> {
    pub(super) fn frame(&self) -> &Frame<T> {
        &self.0.frame
    }

    pub(super) fn balanced(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.0.grounded(Balance::Balanced(grounding))
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
pub(super) struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
    surplus: RowSurplus,
}

/// How a row's `m_ii − Σ_j |m_ij|` reads as surplus.
enum RowSurplus {
    /// Unscaled: inside the band its entries per row allow, it is rounding and reads zero.
    Banded(Vec<usize>),
    /// Scaled: the scaling's residue, which no row bound tells from a real surplus, deficits clamped to zero.
    Residue,
}

impl<T: Real> RowSums<T> {
    /// The surplus `M`'s own rows show.
    fn surplus(self) -> Grounding<T> {
        let rows = self.diagonal.iter().zip(&self.abs);
        let surplus = match &self.surplus {
            RowSurplus::Banded(degrees) => (rows.zip(degrees))
                .map(|((&d, &s), &degree)| scaling::unscaled_surplus(d, s, degree))
                .collect(),
            RowSurplus::Residue => rows.map(|(&d, &s)| (d - s).max(T::zero())).collect(),
        };
        Grounding::of(surplus)
    }
}
