//! Each component reduced on its own; folded Grams share one column pass.

use std::cell::LazyCell;
use std::sync::Arc;

use num_traits::PrimInt;

use super::component::{
    Balance, Component, Frame, FromCsrError, Grounding, OffDiagonal, ReduceError, Sgdd,
};
use super::congruence::Congruence;
use super::decomposition::{
    assemble_bipartite, assemble_bipartite_gram, assemble_general, assemble_signed, of_csr,
    of_upper, BipartiteComponent, Cells, Extracted, Partitioned,
};
use super::null_vectors::null_vectors;
use super::scaling::{self, Existence, Scaling, ScalingPolicy};
use super::signature;
use crate::input::{
    BipartiteGram, BipartiteLaplacian, Canonical, Csr, FactorWidth2, IntoColumns, Laplacian,
    SignedGraph, UpperCanonical, UpperCsr,
};
use crate::storage::{Rows, UpperAdjacency};
use crate::{diagonal_in_range, in_range, narrow, widen, Real};

impl<T: Real, I: PrimInt> Partitioned<Canonical<'_, T, I>> {
    /// Scales and signs each component's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy) -> Result<Sgdd<T>, ReduceError> {
        Ok(self
            .extract(|cells| {
                let a = SymmetricBlock::from(cells);
                if a.is_dead() {
                    return Ok(None);
                }
                let scaling = a.scale(policy)?;
                let component = match a.sign(scaling, policy)? {
                    Signed::Folded(m, rows) => {
                        let grounding = rows.folded_surplus(&m.off_diagonal);
                        m.balanced(grounding)?
                    }
                    Signed::Frustrated(component) => component,
                };
                Ok(Some(component))
            })?
            .into_sgdd())
    }
}

impl<T: Real> Sgdd<T> {
    /// Checks symmetry, then scales each component to dominance unless it already is.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_csr(Canonical::of(csr)?)?.reduce(&policy)?)
    }

    /// As [`from_csr`](Self::from_csr), the triangle symmetric by storage: no mirror is checked.
    pub fn from_upper_csr<I: PrimInt>(
        csr: UpperCsr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_upper(UpperCanonical::of(csr)?)?.reduce(&policy)?)
    }

    /// `A = V Vᵀ`: scales, folds or covers each component as general, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        v: FactorWidth2<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let blocks = assemble_general(v)?.extract(|cells| {
            let a = SymmetricBlock::from(cells);
            Ok((!a.is_dead()).then_some(a))
        })?;
        reduce_gram_blocks(v, blocks, &policy)
    }

    /// `A = V W Vᵀ` on two disjoint sides `[D_l C; Cᵀ D_r]`, as [`from_gram`](Self::from_gram).
    pub fn from_bipartite_gram<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        // Incidence on both sides folds by shape, so `policy` has nothing to scale.
        if let Some(laplacian) = gram.laplacian() {
            return folded_bipartite(laplacian);
        }
        let (c, diagonal) = assemble_bipartite_gram(gram)?;
        let blocks = c.extract(|component| {
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
            let diagonal = vertices.iter().map(|&v| diagonal[v as usize]).collect();
            let a = SymmetricBlock::new(vertices, off_diagonal, diagonal);
            Ok((!a.is_dead()).then_some(a))
        })?;
        drop(diagonal);
        reduce_gram_blocks(gram, blocks, &policy)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Laplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant with surplus `g` by shape: a component floats exactly when its ground is zero.
    fn try_from(graph: Laplacian<'a, T, I>) -> Result<Self, Self::Error> {
        let ground = graph.ground();
        Ok(assemble_general(graph)?
            .extract(|cells| {
                let Cells {
                    vertices,
                    adjacency,
                    diagonal,
                } = cells;
                let gather =
                    |g: &[T]| Grounding::of(vertices.iter().map(|&v| g[v as usize]).collect());
                let grounding = ground.map_or_else(Grounding::ungrounded, gather);
                let g = |i: usize| grounding.surplus().map_or(T::zero(), |g| g[i]);
                // A lone vertex has no edge, so only a ground makes it a component.
                if vertices.len() == 1 && diagonal[0] + g(0) == T::zero() {
                    return Ok(None);
                }
                // `A`'s diagonal `d + g`, summed in `f64`, and each ground like any entry.
                let out_of_range = |i: usize| {
                    !in_range(g(i))
                        || !diagonal_in_range(narrow::<T>(widen(diagonal[i]) + widen(g(i))))
                };
                if let Some(i) = (0..vertices.len()).find(|&i| out_of_range(i)) {
                    let vertex = vertices[i];
                    return Err(ReduceError::OutOfRange { vertex });
                }
                let off_diagonal = OffDiagonal::General(adjacency);
                let m = Congruent::new(vertices, off_diagonal, Congruence::Identity);
                m.balanced(grounding).map(Some)
            })?
            .into_sgdd())
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<SignedGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
    fn try_from(graph: SignedGraph<'a, T, I>) -> Result<Self, Self::Error> {
        let (a, surplus) = assemble_signed(graph)?;
        Ok(a.extract(|cells| {
            let Cells {
                vertices,
                adjacency,
                diagonal,
            } = cells;
            if diagonal == [T::zero()] {
                return Ok(None);
            }
            if let Some(i) = diagonal.iter().position(|&d| !diagonal_in_range(d)) {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let grounding = Grounding::of(vertices.iter().map(|&v| surplus[v as usize]).collect());
            let off_diagonal = OffDiagonal::General(adjacency);
            let component = match signature::folding_signs(&off_diagonal) {
                Some(signs) => Congruent::new(vertices, off_diagonal, signs).balanced(grounding),
                None => Congruent::new(vertices, off_diagonal, Congruence::Identity)
                    .frustrated(grounding),
            };
            component.map(Some)
        })?
        .into_sgdd())
    }
}

/// Scaled by `V`'s null vectors on demand; folded blocks drop their rows for one pass over `V`.
fn reduce_gram_blocks<T: Real>(
    v: impl IntoColumns<T>,
    blocks: Extracted<SymmetricBlock<T>>,
    policy: &ScalingPolicy,
) -> Result<Sgdd<T>, ReduceError> {
    let Extracted {
        partition,
        live,
        dead,
    } = blocks;
    // Built by the first block not already dominant, so a dominant input reads no extra pass.
    let nulls = LazyCell::new(|| null_vectors(v, partition.components()));
    let (mut folded, mut frustrated) = (Vec::new(), Vec::new());
    for a in live {
        let scaling = a.scale_gram(policy, &nulls)?;
        match a.sign(scaling, policy)? {
            Signed::Folded(m, _) => folded.push(m),
            Signed::Frustrated(component) => frustrated.push(component),
        }
    }
    let mut components = ground_or_float(v, folded)?;
    components.extend(frustrated);
    Ok(Sgdd::new(partition.n(), components, dead))
}

/// `M`, a bipartite Laplacian, with the kept side negated folding `A` into it; a lone vertex is dead.
fn folded_bipartite<T: Real, I: PrimInt>(
    m: BipartiteLaplacian<'_, T, I>,
) -> Result<Sgdd<T>, ReduceError> {
    Ok(assemble_bipartite(m)?
        .extract(|component| {
            let BipartiteComponent::Sides(vertices, cross) = component else {
                return Ok(None);
            };
            let signs = Congruence::NegateFrom(cross.n_rows());
            let off_diagonal = OffDiagonal::Bipartite(Arc::new(cross));
            if let Some(i) =
                (off_diagonal.abs_row_sums().iter()).position(|&d| !diagonal_in_range(d))
            {
                let vertex = vertices[i];
                return Err(ReduceError::OutOfRange { vertex });
            }
            let m = Congruent {
                frame: Frame::new(vertices, signs),
                off_diagonal,
            };
            m.balanced(Grounding::ungrounded()).map(Some)
        })?
        .into_sgdd())
}

impl<T: Real> Extracted<Component<T>> {
    fn into_sgdd(self) -> Sgdd<T> {
        Sgdd::new(self.partition.n(), self.live, self.dead)
    }
}

/// One component `A_c` as the input gave it, with its rows' `Σ_j |a_ij|` and entry counts.
struct SymmetricBlock<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
    /// Summed in `f64`: a long `f32` row is not undercounted.
    abs_row_sums: Vec<T>,
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
            off_diagonal,
            diagonal,
        }
    }

    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    /// Rows prove nothing: SCI past the row test.
    fn scale(&self, policy: &ScalingPolicy) -> Result<Scaling<T>, ReduceError> {
        self.scaling(policy, Existence::Unknown, || self.relax(policy))
    }

    /// SCI on general rows, the reduced CG on a bipartite block.
    fn relax(&self, policy: &ScalingPolicy) -> Option<scaling::Relaxation<T>> {
        match &self.off_diagonal {
            OffDiagonal::General(adjacency) => scaling::sci(adjacency, &self.diagonal, policy),
            OffDiagonal::Bipartite(cross) => scaling::reduced_cg(cross, &self.diagonal, policy),
        }
    }

    /// `V` proves a scaling exists; a zero diagonal beside a cell is its terms underflowing.
    fn scale_gram(
        &self,
        policy: &ScalingPolicy,
        nulls: &LazyCell<Vec<f64>, impl FnOnce() -> Vec<f64>>,
    ) -> Result<Scaling<T>, ReduceError> {
        if let Some(i) = self.diagonal.iter().position(|&d| d == T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        self.scaling(policy, Existence::Guaranteed, || {
            let u: Vec<f64> = (self.vertices.iter()).map(|&v| nulls[v as usize]).collect();
            (u.iter().all(|x| x.is_finite()))
                .then(|| scaling::null_scaling(&u, &self.diagonal))
                .flatten()
                .or_else(|| self.relax(policy))
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
        if let Some(i) = self.diagonal.iter().position(|&d| d <= T::zero()) {
            let vertex = self.vertices[i];
            return Err(ReduceError::NonpositiveDiagonal { vertex });
        }
        let vertex = self.vertices[0];
        let mut rows = self.diagonal.iter().zip(&self.abs_row_sums);
        if rows.all(|(&d, &s)| policy.dominates(d, s)) {
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
                Ok(Signed::Folded(m, rows))
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
                };
                Ok((
                    Congruent::new(self.vertices, self.off_diagonal, signs),
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
                let deficit = clamped_deficit(&diagonal, &abs);
                let report = policy.report(&relaxation, existence, deficit);
                let m = Congruent {
                    frame: Frame::new(self.vertices, Congruence::Scaled(factors, report)),
                    off_diagonal,
                };
                let rows = RowSums { diagonal, abs };
                Ok((m, rows))
            }
        }
    }
}

impl<T: Real> OffDiagonal<T> {
    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    fn abs_row_sums(&self) -> Vec<T> {
        match self {
            Self::General(adjacency) => adjacency.abs_row_sums(),
            Self::Bipartite(cross) => {
                let (eliminated, kept) = cross.abs_sums();
                eliminated.into_iter().chain(kept).map(narrow).collect()
            }
        }
    }

    fn abs_total(&self) -> f64 {
        match self {
            Self::General(adjacency) => adjacency.abs_total(),
            Self::Bipartite(c) => 2.0 * c.abs_total(),
        }
    }

    /// `S X S`: a sign moves no magnitude, so no entry leaves range.
    fn signed(self, s: &Congruence<T>) -> Self {
        let Some(s) = s.factors(self.n()) else {
            return self;
        };
        match self {
            Self::General(adjacency) => Self::General(adjacency.signed(&s)),
            Self::Bipartite(cross) => {
                let (eliminated, kept) = s.split_at(cross.n_rows());
                Self::Bipartite(Arc::new(
                    Arc::unwrap_or_clone(cross).signed(eliminated, kept),
                ))
            }
        }
    }

    /// `F X F` with the new row sums from the same pass; `None` where an entry leaves range.
    fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        Some(match self {
            Self::General(adjacency) => {
                let (adjacency, sums) = adjacency.scaled_abs_row_sums(f)?;
                (Self::General(adjacency), sums)
            }
            Self::Bipartite(cross) => {
                let (eliminated, kept) = f.split_at(cross.n_rows());
                let (cross, rows, cols) =
                    Arc::unwrap_or_clone(cross).scaled_abs_sums(eliminated, kept)?;
                let sums = rows.into_iter().chain(cols).map(narrow).collect();
                (Self::Bipartite(Arc::new(cross)), sums)
            }
        })
    }
}

/// Only a folded `M` can still take a surplus from `V`.
enum Signed<T> {
    Folded(Congruent<T>, RowSums<T>),
    Frustrated(Component<T>),
}

/// `M = F · A_c · F` over its frame, its surplus still to be decided.
struct Congruent<T> {
    frame: Frame<T>,
    off_diagonal: OffDiagonal<T>,
}

impl<T: Real> Congruent<T> {
    /// `S A_c S`, unscaled.
    fn new(vertices: Box<[u32]>, off_diagonal: OffDiagonal<T>, signs: Congruence<T>) -> Self {
        Self {
            off_diagonal: off_diagonal.signed(&signs),
            frame: Frame::new(vertices, signs),
        }
    }

    fn frustrated(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.grounded(Balance::Frustrated(grounding))
    }

    fn balanced(self, grounding: Grounding<T>) -> Result<Component<T>, ReduceError> {
        self.grounded(Balance::Balanced(grounding))
    }

    fn grounded(self, balance: Balance<T>) -> Result<Component<T>, ReduceError> {
        Component::new(self.frame, self.off_diagonal, balance)
    }
}

/// `M`'s diagonal and `Σ_j |m_ij|`: the surplus when no shape or `V` gives it.
struct RowSums<T> {
    diagonal: Vec<T>,
    abs: Vec<T>,
}

impl<T: Real> RowSums<T> {
    /// A folded `M`'s row sums are `M1`, so `1ᵀM1 = tr M − Σ|offdiag(M)|`, both compensated.
    fn folded_surplus(self, off_diagonal: &OffDiagonal<T>) -> Grounding<T> {
        let trace = crate::compensated_sum(self.diagonal.iter().map(|&d| widen(d)));
        let ones = trace - off_diagonal.abs_total();
        match ones <= SINGULAR_BAND * widen(T::epsilon()) * trace {
            true => Grounding::ungrounded(),
            false => self.surplus(),
        }
    }

    /// Each row's residue, deficits clamped to zero.
    fn surplus(self) -> Grounding<T> {
        let rows = self.diagonal.iter().zip(&self.abs);
        Grounding::of(rows.map(|(&d, &s)| (d - s).max(T::zero())).collect())
    }
}

/// Floats each folded component as [`RowSums::folded_surplus`] does, else grounds it on `s = M·1`.
fn ground_or_float<T: Real>(
    v: impl IntoColumns<T>,
    components: Vec<Congruent<T>>,
) -> Result<Vec<Component<T>>, ReduceError> {
    if components.is_empty() {
        return Ok(Vec::new());
    }
    // By vertex: components are disjoint, so one slot per vertex holds its component, `f_i` and surplus.
    let mut component_of = vec![u32::MAX; v.n()];
    let mut factor = vec![0.0; v.n()];
    for (c, component) in components.iter().enumerate() {
        let congruence = component.frame.congruence();
        for (slot, &vertex) in component.frame.vertices().iter().enumerate() {
            component_of[vertex as usize] = c as u32;
            factor[vertex as usize] = widen(congruence.factor(slot));
        }
    }
    let mut surplus = vec![0.0f64; v.n()];
    // `ε·1ᵀM1` and `ε·tr M` per component.
    let (mut ones, mut trace) = (
        vec![0.0f64; components.len()],
        vec![0.0f64; components.len()],
    );
    let epsilon = widen(T::epsilon());
    v.for_each_column(
        #[inline(always)]
        |column| {
            if column.weight.is_zero() {
                return;
            }
            let w = |x: f64, y: f64| column.weight.bilinear(x, y);
            let end = |i: u32, v_i: T| match component_of[i as usize] {
                u32::MAX => None,
                c => Some((i as usize, c as usize, factor[i as usize] * widen(v_i))),
            };
            match (end(column.a, column.v_a), end(column.b, column.v_b)) {
                (Some((a, c, u_a)), Some((b, c_b, u_b))) if c == c_b => {
                    let p = u_a + u_b;
                    surplus[a] += w(u_a, p);
                    surplus[b] += w(u_b, p);
                    ones[c] += w(epsilon * p, p);
                    trace[c] += w(epsilon * u_a, u_a) + w(epsilon * u_b, u_b);
                }
                // A column straddling components keeps only its own end's square.
                (a, b) => {
                    for (i, c, u) in [a, b].into_iter().flatten() {
                        surplus[i] += w(u, u);
                        ones[c] += w(epsilon * u, u);
                        trace[c] += w(epsilon * u, u);
                    }
                }
            }
        },
    );
    drop((component_of, factor));
    let decided = components.into_iter().zip(ones).zip(trace);
    let decide = |((component, ones), trace): ((Congruent<T>, f64), f64)| {
        if ones <= SINGULAR_BAND * epsilon * trace {
            return component.balanced(Grounding::ungrounded());
        }
        let vertices = component.frame.vertices();
        let surplus: Vec<T> = (vertices.iter())
            .map(|&v| narrow(surplus[v as usize].max(0.0)))
            .collect();
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

/// `max_i (Σ_j|m_ij| − m_ii)/m_ii` past the entries' rounding: what clamping adds.
fn clamped_deficit<T: Real>(diagonal: &[T], abs_row_sums: &[T]) -> f64 {
    let band = SINGULAR_BAND * widen(T::epsilon());
    (diagonal.iter().zip(abs_row_sums))
        .map(|(&d, &s)| (widen(s) - widen(d)) / widen(d))
        .filter(|&deficit| deficit > band)
        .fold(0.0, f64::max)
}

/// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh), so `M` floats.
const SINGULAR_BAND: f64 = 5.0;
