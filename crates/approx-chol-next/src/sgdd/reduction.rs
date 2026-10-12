//! Each component reduced on its own; folded Grams share one column pass.

use std::cell::LazyCell;
use std::sync::Arc;

use num_traits::PrimInt;

use super::congruence::Signature;
use super::decomposition::{
    assemble_bipartite, assemble_general, BipartiteComponent, Cells, DirectSum, Partitioned,
};
use super::null_vectors::{LazyNullVectors, NullVectors};
use super::{scaling, signature, Existence};
use super::{
    Balance, Congruence, Frame, Grounding, OffDiagonal, ReduceError, ScalingPolicy, ScalingReport,
    Sgdd,
};
use crate::input::{BipartiteGram, Canonical, IntoColumns};
use crate::storage::{in_range, narrow, widen, Rows, UpperAdjacency};
use crate::Real;

impl<T: Real, I: PrimInt> Partitioned<Canonical<'_, T, I>> {
    /// Scales and signs each component's `A_c`, closing its surplus from the same rows.
    pub(super) fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let sum = self.extract(|cells| {
            let a = SymmetricBlock::from(cells);
            if a.is_dead() {
                return Ok(None);
            }
            let component = match a.scale(policy)?.sign()? {
                Signed::Folded(m, rows) => m.balanced(rows.surplus())?,
                Signed::Frustrated(component) => component,
            };
            Ok(Some(component))
        })?;
        Ok(sum.into())
    }
}

/// `⊕ A_c` of `A = V W Vᵀ`, its assembly already freed, with the `V` its null vectors and folded surplus read.
pub(super) struct GramSum<T, V> {
    v: V,
    sum: DirectSum<SymmetricBlock<T>>,
}

impl<T: Real, V: IntoColumns<T>> GramSum<T, V> {
    /// `A = V Vᵀ` as general components.
    pub(super) fn general(v: V) -> Result<Self, ReduceError> {
        let sum = assemble_general(v)?.extract(|cells| {
            let a = SymmetricBlock::from(cells);
            Ok((!a.is_dead()).then_some(a))
        })?;
        Ok(Self { v, sum })
    }

    /// As [`Partitioned::reduce`], scaled by `V`'s null vectors on demand; a folded component drops its rows for the one column pass that grounds or floats them all.
    pub(super) fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        let Self { v, sum } = self;
        let sum = sum.reduce_all(|partition, blocks| {
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
            Ok(components)
        })?;
        Ok(sum.into())
    }
}

impl<'g, T: Real, I: PrimInt> GramSum<T, BipartiteGram<'g, T, I>> {
    /// `[D_l C; Cᵀ D_r]`, each component's diagonal read off the sides.
    pub(super) fn bipartite(gram: BipartiteGram<'g, T, I>) -> Result<Self, ReduceError> {
        let c = assemble_bipartite(gram)?;
        // The sides' diagonals are built on first read: after the assembly's peak.
        let (left, right) = gram.diagonals();
        let side_diagonal = |v: u32| match (v as usize).checked_sub(left.len()) {
            None => left[v as usize],
            Some(j) => right[j],
        };
        let sum = c.extract(|component| {
            let (vertices, off_diagonal) = match component {
                BipartiteComponent::Lone(v) => {
                    let empty = Rows::new(vec![0, 0], Vec::new(), Vec::new(), 1);
                    let empty = OffDiagonal::General(UpperAdjacency::new(empty));
                    (Box::from([v]), empty)
                }
                BipartiteComponent::Sides(vertices, cross) => {
                    (vertices, OffDiagonal::Bipartite(Arc::new(cross)))
                }
            };
            let diagonal = vertices.iter().map(|&v| side_diagonal(v)).collect();
            let a = SymmetricBlock::new(vertices, off_diagonal, diagonal);
            Ok((!a.is_dead()).then_some(a))
        })?;
        Ok(Self { v: gram, sum })
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

impl<T: Real> From<Cells<T>> for SymmetricBlock<T> {
    fn from(cells: Cells<T>) -> Self {
        let off_diagonal = OffDiagonal::General(cells.adjacency);
        Self::new(cells.vertices, off_diagonal, cells.diagonal)
    }
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
        match signature::folding_signs(&self.a.off_diagonal) {
            Some(signs) => {
                let (m, rows) = self.congruent(signs)?;
                Ok(Signed::Folded(Folded::new(m), rows))
            }
            None => {
                let (m, rows) = self.congruent(Signature::identity())?;
                Ok(Signed::Frustrated(m.frustrated(rows.surplus())?))
            }
        }
    }

    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the report; a frustrated `A_c` takes `S = I`.
    fn congruent(self, signs: Signature<T>) -> Result<(Congruent<T>, RowSums<T>), ReduceError> {
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
                let factors = signs.times(&relaxation.scales);
                let (off_diagonal, abs) = (a.off_diagonal.scaled_abs_row_sums(&factors))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal: Vec<T> = (a.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let deficit = scaling::clamped_deficit(&diagonal, &abs, &a.degrees);
                let congruence = Congruence::Scaled(factors, certify.report(&relaxation, deficit));
                let m = Congruent {
                    frame: Frame::new(a.vertices, congruence),
                    off_diagonal,
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

/// `M = F · A_c · F` over its frame, its surplus still to be decided.
pub(super) struct Congruent<T> {
    pub(super) frame: Frame<T>,
    pub(super) off_diagonal: OffDiagonal<T>,
}

impl<T: Real> Congruent<T> {
    /// `F A_c F` unscaled, so nothing is reported.
    pub(super) fn new(
        vertices: Box<[u32]>,
        off_diagonal: OffDiagonal<T>,
        signs: Signature<T>,
    ) -> Result<Self, ReduceError> {
        let vertex = vertices[0];
        let off_diagonal =
            (signs.transform(off_diagonal)).ok_or(ReduceError::Unscalable { vertex })?;
        Ok(Self {
            frame: Frame::new(vertices, Congruence::Signed(signs)),
            off_diagonal,
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
        Ok(Component::new(self, balance))
    }
}

/// `M` folded to a Z-matrix by a found signature or by its shape: only it grounds as balanced.
pub(super) struct Folded<T>(Congruent<T>);

impl<T: Real> Folded<T> {
    /// Panics unless `M` is a Z-matrix.
    pub(super) fn new(m: Congruent<T>) -> Self {
        assert!(m.off_diagonal.nonpositive(), "a folded M is a Z-matrix");
        Self(m)
    }

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

/// Floats each folded component when `1ᵀM1 ≤ 5ε·tr M`, else grounds it on `s = M·1`, both summed by column as assembled rows cancel them: with `f = F·1` and `p_k = f_a v_a + f_b v_b`, `s_a = Σ_k bilinear_k(f_a v_a, p_k)` clamped at zero, `1ᵀM1 = Σ_k bilinear_k(p_k, p_k)` and `tr M = Σ_k [bilinear_k(f_a v_a, f_a v_a) + bilinear_k(f_b v_b, f_b v_b)]`, `ε` that of `T`; sums run in `f64` in column order over nonzero weights, the certificate's two sides as `ε·term`, an exact scaling under which neither overflows, each end's component and slot are read off the components' own vertices, an end at a dead vertex or in a frustrated component being in none; a column straddling components keeps only its own end's square.
fn ground_or_float<T: Real>(
    v: impl IntoColumns<T>,
    components: Vec<Folded<T>>,
) -> Result<Vec<Component<T>>, ReduceError> {
    if components.is_empty() {
        return Ok(Vec::new());
    }
    let mut ends = vec![None; v.n()];
    for (c, component) in components.iter().enumerate() {
        let congruence = component.frame().congruence();
        for (slot, &vertex) in component.frame().vertices().iter().enumerate() {
            ends[vertex as usize] = Some(End {
                component: c,
                slot,
                factor: widen(congruence.factor(slot)),
            });
        }
    }
    let mut surplus: Vec<Vec<f64>> = (components.iter())
        .map(|component| vec![0.0; component.frame().vertices().len()])
        .collect();
    let mut certificates = vec![Certificate::default(); components.len()];
    let epsilon = widen(T::epsilon());
    v.for_each_column(
        #[inline(always)]
        |column| {
            if column.weight.is_zero() {
                return;
            }
            let w = |x: f64, y: f64| column.weight.bilinear(x, y);
            let a = ends[column.a as usize].map(|end| (end, end.factor * widen(column.v_a)));
            let b = ends[column.b as usize].map(|end| (end, end.factor * widen(column.v_b)));
            match (a, b) {
                (Some((a, u_a)), Some((b, u_b))) if a.component == b.component => {
                    let p = u_a + u_b;
                    surplus[a.component][a.slot] += w(u_a, p);
                    surplus[b.component][b.slot] += w(u_b, p);
                    let certificate = &mut certificates[a.component];
                    certificate.ones += w(epsilon * p, p);
                    certificate.trace += w(epsilon * u_a, u_a) + w(epsilon * u_b, u_b);
                }
                (a, b) => {
                    for (end, u) in [a, b].into_iter().flatten() {
                        surplus[end.component][end.slot] += w(u, u);
                        let certificate = &mut certificates[end.component];
                        certificate.ones += w(epsilon * u, u);
                        certificate.trace += w(epsilon * u, u);
                    }
                }
            }
        },
    );
    let decided = components.into_iter().zip(surplus).zip(certificates);
    let decide = |((component, surplus), certificate): ((Folded<T>, Vec<f64>), Certificate)| {
        if certificate.ones <= 5.0 * epsilon * certificate.trace {
            return component.balanced(Grounding::ungrounded());
        }
        let vertices = component.frame().vertices();
        let surplus: Vec<T> = surplus.into_iter().map(|s| narrow(s.max(0.0))).collect();
        if let Some(slot) = surplus.iter().position(|&s| !in_range(s)) {
            let vertex = vertices[slot];
            return Err(ReduceError::OutOfRange { vertex });
        }
        // Unproven, a surplus that vanishes in `T` is never a float.
        let vertex = vertices[0];
        match Grounding::of(surplus) {
            grounding if grounding.is_grounded() => component.balanced(grounding),
            _ => Err(ReduceError::Unscalable { vertex }),
        }
    };
    decided.map(decide).collect()
}

/// Where a column's end sits once reduced, and its `f_i`.
#[derive(Clone, Copy)]
struct End {
    component: usize,
    slot: usize,
    factor: f64,
}

/// `ε·1ᵀM1` and `ε·tr M` of one component, summed by column.
#[derive(Clone, Copy, Default)]
struct Certificate {
    ones: f64,
    trace: f64,
}

/// One component of `A`, reduced: `M = F · A_c · F` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Component<T = f64> {
    frame: Arc<Frame<T>>,
    off_diagonal: OffDiagonal<T>,
    balance: Balance<T>,
}

impl<T: Real> Component<T> {
    /// Panics unless one row and one surplus per vertex.
    fn new(m: Congruent<T>, balance: Balance<T>) -> Self {
        let n = m.frame.vertices().len();
        assert_eq!(m.off_diagonal.n(), n, "one row per vertex");
        assert!(balance.grounding().fits(n), "one surplus per row");
        Self {
            frame: Arc::new(m.frame),
            off_diagonal: m.off_diagonal,
            balance,
        }
    }

    /// The component's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
    }

    /// Present iff this component's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.frame.congruence().report()
    }

    /// Shared with every factor of this component.
    pub(crate) fn frame(&self) -> &Arc<Frame<T>> {
        &self.frame
    }

    /// Whether `A_c` is proven singular, so a solve returns `A_c⁺b`.
    pub fn floats(&self) -> bool {
        matches!(&self.balance, Balance::Balanced(grounding) if !grounding.is_grounded())
    }

    /// `M`: its off-diagonal and its balance.
    pub(crate) fn m(&self) -> (&OffDiagonal<T>, &Balance<T>) {
        (&self.off_diagonal, &self.balance)
    }
}
