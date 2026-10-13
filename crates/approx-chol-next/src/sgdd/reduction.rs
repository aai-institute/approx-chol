//! Each component reduced on its own; folded Grams share one column pass.

use std::cell::LazyCell;
use std::sync::Arc;

use num_traits::PrimInt;

use super::component::{Balance, Component, Frame, Grounding, OffDiagonal, ReduceError, Sgdd};
use super::congruence::Congruence;
use super::decomposition::{
    assemble_bipartite, assemble_general, assemble_signed, BipartiteComponent, Cells, Partition,
    Partitioned,
};
use super::null_vectors::NullVectors;
use super::scaling::{self, Existence, Scaling, ScalingPolicy};
use super::signature;
use crate::input::{
    BipartiteGram, BipartiteGraph, BipartiteLaplacian, Canonical, IntoColumns, Laplacian,
    SignedGraph,
};
use crate::storage::{diagonal_in_range, in_range, narrow, widen, Rows, UpperAdjacency};
use crate::Real;

impl<T: Real, I: PrimInt> Partitioned<Canonical<'_, T, I>> {
    /// Scales and signs each component's `A_c`, closing its surplus from the same rows.
    pub(super) fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        self.extract(|cells| {
            let a = SymmetricBlock::from(cells);
            if a.is_dead() {
                return Ok(None);
            }
            let scaling = a.scale(policy)?;
            let component = match a.sign(scaling, policy)? {
                Signed::Folded(m, rows) => m.balanced(rows.surplus())?,
                Signed::Frustrated(component) => component,
            };
            Ok(Some(component))
        })
    }
}

/// `A = V Vᵀ` as general components.
pub(super) fn reduce_gram<T: Real>(
    v: impl IntoColumns<T>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    assemble_general(v)?.extract_all(
        |cells| {
            let a = SymmetricBlock::from(cells);
            Ok((!a.is_dead()).then_some(a))
        },
        |partition, blocks| reduce_gram_blocks(v, partition, blocks, policy),
    )
}

/// `[D_l C; Cᵀ D_r]`, each component's diagonal read off the sides.
pub(super) fn reduce_bipartite_gram<T: Real, I: PrimInt>(
    gram: BipartiteGram<'_, T, I>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    let c = assemble_bipartite(gram)?;
    // Built on first read, so after the assembly's peak.
    let (left, right) = gram.diagonals();
    let side_diagonal = |v: u32| match (v as usize).checked_sub(left.len()) {
        None => left[v as usize],
        Some(j) => right[j],
    };
    c.extract_all(
        |component| {
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
        },
        |partition, blocks| reduce_gram_blocks(gram, partition, blocks, policy),
    )
}

/// Scaled by `V`'s null vectors on demand; folded blocks drop their rows for one pass over `V`.
fn reduce_gram_blocks<T: Real>(
    v: impl IntoColumns<T>,
    partition: &Partition,
    blocks: Vec<SymmetricBlock<T>>,
    policy: &ScalingPolicy,
) -> Result<Vec<Component<T>>, ReduceError> {
    // Built by the first block not already dominant, so a dominant input reads no extra pass.
    let nulls = LazyCell::new(|| NullVectors::of(v, partition.components()));
    let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
    for a in blocks {
        let scaling = a.scale_gram(policy, &nulls)?;
        match a.sign(scaling, policy)? {
            Signed::Folded(m, _) => folded.push(m),
            Signed::Frustrated(component) => frustrated.push(component),
        }
    }
    let mut components = ground_or_float(v, folded)?;
    components.extend(frustrated);
    Ok(components)
}

/// Dominant with surplus `g` by shape: a component floats exactly when its ground is zero.
pub(super) fn reduce_laplacian<T: Real, I: PrimInt>(
    graph: Laplacian<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    let ground = graph.ground();
    assemble_general(graph)?.extract(|cells| {
        let Cells {
            vertices,
            adjacency,
            diagonal,
        } = cells;
        let gather = |g: &[T]| Grounding::of(vertices.iter().map(|&v| g[v as usize]).collect());
        let grounding = ground.map_or_else(Grounding::ungrounded, gather);
        let g = |i: usize| grounding.surplus().map_or(T::zero(), |g| g[i]);
        // A lone vertex has no edge, so only a ground makes it a component.
        if vertices.len() == 1 && diagonal[0] + g(0) == T::zero() {
            return Ok(None);
        }
        // `A`'s diagonal `d + g`, summed in `f64`, and each ground like any entry.
        let out_of_range = |i: usize| {
            !in_range(g(i)) || !diagonal_in_range(narrow::<T>(widen(diagonal[i]) + widen(g(i))))
        };
        if let Some(i) = (0..vertices.len()).find(|&i| out_of_range(i)) {
            let vertex = vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let off_diagonal = OffDiagonal::General(adjacency);
        let m = Congruent::new(vertices, off_diagonal, Congruence::Identity)?;
        Folded::new(m).balanced(grounding).map(Some)
    })
}

/// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
pub(super) fn reduce_signed<T: Real, I: PrimInt>(
    graph: SignedGraph<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    let (a, surplus) = assemble_signed(graph)?;
    a.extract(|cells| {
        let Cells {
            vertices,
            adjacency,
            diagonal,
        } = cells;
        if diagonal == [T::zero()] {
            return Ok(None);
        }
        // A zero diagonal beside a cell is its terms underflowing.
        let out_of_range = |&d: &T| d == T::zero() || !diagonal_in_range(d);
        if let Some(i) = diagonal.iter().position(out_of_range) {
            let vertex = vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let grounding = Grounding::of(vertices.iter().map(|&v| surplus[v as usize]).collect());
        let off_diagonal = OffDiagonal::General(adjacency);
        let component = match signature::folding_signs(&off_diagonal) {
            Some(signs) => {
                Folded::new(Congruent::new(vertices, off_diagonal, signs)?).balanced(grounding)
            }
            None => (Congruent::new(vertices, off_diagonal, Congruence::Identity)?)
                .frustrated(grounding),
        };
        component.map(Some)
    })
}

/// Negating one side folds it into a bipartite Laplacian: every component floats untested.
pub(super) fn reduce_bipartite_graph<T: Real, I: PrimInt>(
    graph: BipartiteGraph<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    folded_bipartite(graph.laplacian(), Congruence::NegateFrom)
}

/// A Z-matrix with zero surplus by shape: every component floats untested.
pub(super) fn reduce_bipartite_laplacian<T: Real, I: PrimInt>(
    graph: BipartiteLaplacian<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    folded_bipartite(graph, |_| Congruence::Identity)
}

/// `M`, a bipartite Laplacian, with `signs` folding `A` into it; a lone vertex is dead.
fn folded_bipartite<T: Real, I: PrimInt>(
    m: BipartiteLaplacian<'_, T, I>,
    signs: impl Fn(usize) -> Congruence<T>,
) -> Result<Sgdd<T>, ReduceError> {
    assemble_bipartite(m)?.extract(|component| {
        let BipartiteComponent::Sides(vertices, cross) = component else {
            return Ok(None);
        };
        if let Some(i) = (cross.abs_row_sums().iter()).position(|&d| !diagonal_in_range(d)) {
            let vertex = vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let signs = signs(cross.n_eliminated());
        let m = Congruent {
            frame: Frame::new(vertices, signs),
            off_diagonal: OffDiagonal::Bipartite(Arc::new(cross)),
        };
        Folded::new(m).balanced(Grounding::ungrounded()).map(Some)
    })
}

/// One component `A_c` as the input gave it, with its rows' `Σ_j |a_ij|` and entry counts.
pub(super) struct SymmetricBlock<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
    /// Summed in `f64`: a long `f32` row is not undercounted.
    abs_row_sums: Vec<T>,
    /// The rounding a row sum can carry.
    degrees: Vec<usize>,
}

impl<T: Real> From<Cells<T>> for SymmetricBlock<T> {
    fn from(cells: Cells<T>) -> Self {
        let off_diagonal = OffDiagonal::General(cells.adjacency);
        Self::new(cells.vertices, off_diagonal, cells.diagonal)
    }
}

impl<T: Real> SymmetricBlock<T> {
    /// Panics unless one vertex and one diagonal per row.
    fn new(vertices: Box<[u32]>, off_diagonal: OffDiagonal<T>, diagonal: Vec<T>) -> Self {
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
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    /// Rows prove nothing: SCI past the row test, or the reduced CG on a bipartite block.
    fn scale(&self, policy: &ScalingPolicy) -> Result<Scaling<T>, ReduceError> {
        self.scaling(policy, Existence::Unknown, || match &self.off_diagonal {
            OffDiagonal::General(adjacency) => scaling::sci(adjacency, &self.diagonal, policy),
            OffDiagonal::Bipartite(cross) => scaling::reduced_cg(cross, &self.diagonal, policy),
        })
    }

    /// `V` proves a scaling exists; a zero diagonal beside a cell is its terms underflowing.
    fn scale_gram(
        &self,
        policy: &ScalingPolicy,
        nulls: &LazyCell<NullVectors, impl FnOnce() -> NullVectors>,
    ) -> Result<Scaling<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| d == T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        self.scaling(policy, Existence::Guaranteed, || match &self.off_diagonal {
            OffDiagonal::General(adjacency) => (LazyCell::force(nulls).on(&self.vertices))
                .and_then(|u| scaling::null_scaling(&u, &self.diagonal))
                .or_else(|| scaling::sci(adjacency, &self.diagonal, policy)),
            OffDiagonal::Bipartite(cross) => scaling::reduced_cg(cross, &self.diagonal, policy),
        })
    }

    /// Screens the diagonal and row sums; `relax` runs only past the row test.
    fn scaling(
        &self,
        policy: &ScalingPolicy,
        existence: Existence,
        relax: impl FnOnce() -> Option<scaling::Relaxation<T>>,
    ) -> Result<Scaling<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| !diagonal_in_range(d)) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        if let Some(i) = self
            .abs_row_sums
            .iter()
            .position(|&s| !diagonal_in_range(s))
        {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let vertex = self.vertices[0];
        if self.diagonal.iter().any(|&d| d <= T::zero()) {
            return Err(ReduceError::NotGdd { vertex });
        }
        let mut rows = (self.diagonal.iter().zip(&self.abs_row_sums)).zip(&self.degrees);
        if rows.all(|((&d, &s), &degree)| policy.dominates(d, s, degree)) {
            return Ok(Scaling::AlreadyDominant);
        }
        let relaxation = relax().ok_or(ReduceError::Unscalable { vertex })?;
        Ok(Scaling::Scaled(relaxation, existence))
    }

    /// `M = F A_c F` with `F = S R`; a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn sign(self, scaling: Scaling<T>, policy: &ScalingPolicy) -> Result<Signed<T>, ReduceError> {
        match signature::folding_signs(&self.off_diagonal) {
            Some(signs) => {
                let (m, rows) = self.congruent(signs, scaling, policy)?;
                Ok(Signed::Folded(Folded::new(m), rows))
            }
            None => {
                let (m, rows) = self.congruent(Congruence::Identity, scaling, policy)?;
                Ok(Signed::Frustrated(m.frustrated(rows.surplus())?))
            }
        }
    }

    /// `M` formed once with its rows, which decide the report.
    fn congruent(
        self,
        signs: Congruence<T>,
        scaling: Scaling<T>,
        policy: &ScalingPolicy,
    ) -> Result<(Congruent<T>, RowSums<T>), ReduceError> {
        match scaling {
            Scaling::AlreadyDominant => {
                // Signs keep every magnitude, so `A_c`'s rows are `M`'s.
                let rows = RowSums {
                    diagonal: self.diagonal,
                    abs: self.abs_row_sums,
                    surplus: RowSurplus::Banded(self.degrees),
                };
                Ok((
                    Congruent::new(self.vertices, self.off_diagonal, signs)?,
                    rows,
                ))
            }
            Scaling::Scaled(relaxation, existence) => {
                let vertex = self.vertices[0];
                let factors = signs.times(&relaxation.scales);
                let (off_diagonal, abs) = (self.off_diagonal.scaled_abs_row_sums(&factors))
                    .ok_or(ReduceError::Unscalable { vertex })?;
                let diagonal: Vec<T> = (self.diagonal.iter().zip(&relaxation.scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                let deficit = scaling::clamped_deficit(&diagonal, &abs, &self.degrees);
                let report = policy.report(&relaxation, existence, deficit);
                let m = Congruent {
                    frame: Frame::new(self.vertices, Congruence::Scaled(factors, report)),
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

/// Only a folded `M` can still take a surplus from `V`.
enum Signed<T> {
    Folded(Folded<T>, RowSums<T>),
    Frustrated(Component<T>),
}

/// `M = F · A_c · F` over its frame, its surplus still to be decided.
struct Congruent<T> {
    frame: Frame<T>,
    off_diagonal: OffDiagonal<T>,
}

impl<T: Real> Congruent<T> {
    /// `S A_c S`, unscaled.
    fn new(
        vertices: Box<[u32]>,
        off_diagonal: OffDiagonal<T>,
        signs: Congruence<T>,
    ) -> Result<Self, ReduceError> {
        let vertex = vertices[0];
        let off_diagonal =
            (off_diagonal.congruent(&signs)).ok_or(ReduceError::Unscalable { vertex })?;
        Ok(Self {
            frame: Frame::new(vertices, signs),
            off_diagonal,
        })
    }

    fn frustrated(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.grounded(Balance::Frustrated(grounding))
    }

    fn grounded(self, balance: Balance<T>) -> Result<Component<T>, ReduceError> {
        Component::new(self.frame, self.off_diagonal, balance)
    }
}

/// `M` folded to a Z-matrix: only it grounds as balanced.
struct Folded<T>(Congruent<T>);

impl<T: Real> Folded<T> {
    /// Panics unless `M` is a Z-matrix.
    fn new(m: Congruent<T>) -> Self {
        assert!(m.off_diagonal.nonpositive(), "a folded M is a Z-matrix");
        Self(m)
    }

    fn balanced(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.0.grounded(Balance::Balanced(grounding))
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
    surplus: RowSurplus,
}

/// How a row's `m_ii − Σ_j |m_ij|` reads as surplus.
enum RowSurplus {
    /// Unscaled: inside its rounding band it reads zero.
    Banded(Vec<usize>),
    /// Scaled: the scaling's residue, deficits clamped to zero.
    Residue,
}

impl<T: Real> RowSums<T> {
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

/// Floats each folded component when `1ᵀM1 ≤ 5ε·tr M`, else grounds it on `s = M·1`.
fn ground_or_float<T: Real>(
    v: impl IntoColumns<T>,
    components: Vec<Folded<T>>,
) -> Result<Vec<Component<T>>, ReduceError> {
    if components.is_empty() {
        return Ok(Vec::new());
    }
    let mut ends = vec![None; v.n()];
    for (c, Folded(component)) in components.iter().enumerate() {
        let congruence = component.frame.congruence();
        for (slot, &vertex) in component.frame.vertices().iter().enumerate() {
            ends[vertex as usize] = Some(End {
                component: c,
                slot,
                factor: widen(congruence.factor(slot)),
            });
        }
    }
    let mut surplus: Vec<Vec<f64>> = (components.iter())
        .map(|Folded(component)| vec![0.0; component.frame.vertices().len()])
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
                // A column straddling components keeps only its own end's square.
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
        let vertices = component.0.frame.vertices();
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

/// `ε·1ᵀM1` and `ε·tr M` of one component.
#[derive(Clone, Copy, Default)]
struct Certificate {
    ones: f64,
    trace: f64,
}
