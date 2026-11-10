//! Each block reduced on its own; folded Grams share one column pass.

mod gram;
mod graph;

use num_traits::PrimInt;

use super::block::{Balance, BlockComplement, Congruent, Grounding, IrreducibleBlock};
use super::congruence::Congruence;
use super::decomposition::{
    of_csr, AdjacencyLayout, Biadjacency, Partitioned, RowSums, StrictUpperTriangle, SymmetricBlock,
};
use super::scaling::{self, ScaleVector, ScalingPolicy};
use super::{Reduced, ReducedComplement};
use crate::diagonal_in_range;
use crate::input::{BipartiteFactorWidth2, Canonical, FactorWidth2, Input, Lowered, Side};
use crate::storage::OffDiagonal;
use crate::{argmax, narrow, widen, BuildError, Index, Real, Threads};

impl<T: Real, I: Index> Reduced<T, I> {
    /// The view's irreducible blocks, each reduced; `policy` scales those not already dominant.
    pub(crate) fn reduce<J: PrimInt + Sync>(
        input: Input<'_, T, J>,
        policy: ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        Ok(match input.0 {
            Lowered::Csr(rows) => of_csr(Canonical::of(rows)).reduce(&policy, threads)?,
            Lowered::FactorWidth2(v) => Self::from_factor_width_2(v, &policy, threads)?,
        })
    }

    fn from_factor_width_2<J: PrimInt>(
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        match Biadjacency::of(&v) {
            Some(layout) => Self::from_gram(layout, v, policy, threads),
            None => Self::from_gram(StrictUpperTriangle, v, policy, threads),
        }
    }

    /// An incidence `V` is dominant by shape; values are scaled under `policy`.
    fn from_gram<J: PrimInt, S: AdjacencyLayout>(
        cells: S,
        v: FactorWidth2<'_, T, J>,
        policy: &ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        match v.is_incidence() {
            true => Self::from_incidence(cells, v, threads),
            false => Self::from_values(cells, v, policy, threads),
        }
    }
}

impl<T: Real, I: Index> ReducedComplement<T, I> {
    /// As [`Reduced::reduce`], each block's side opposite `kept` eliminated first.
    pub(crate) fn reduce_onto<J: PrimInt>(
        input: BipartiteFactorWidth2<'_, T, J>,
        kept: Side,
        policy: ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        let range = input.vertices(kept);
        let layout = Biadjacency::onto(&input, kept);
        let whole: Reduced<T, I> = Reduced::from_gram(layout, input.lower(), &policy, threads)?;
        let mut complement = whole.map(threads, |vertices, block| {
            BlockComplement::new(block, vertices[0].index())
        });
        complement.retain_isolated(|v| range.contains(&v.index()));
        complement.numbering = range;
        Ok(complement)
    }
}

impl<T: Real, J: PrimInt + Sync, I: Index> Partitioned<Canonical<'_, T, J>, I> {
    /// Scales and signs each block's `A_c`, closing its surplus from the same rows.
    fn reduce(self, policy: &ScalingPolicy, threads: Threads) -> Result<Reduced<T, I>, BuildError> {
        self.reduced(threads, |vertices, entries, threads| {
            let a = SymmetricBlock { vertices, entries };
            let relax = |a: &SymmetricBlock<'_, T, I>| {
                scaling::relax(
                    &a.entries.off_diagonal,
                    &a.entries.rows.diagonal,
                    policy,
                    threads,
                )
            };
            let block = match a.reduce(policy, relax)? {
                Settling::Pending(folded) => {
                    let band = folded.band();
                    folded.settle(band, &RowSurplus)?
                }
                Settling::Settled(block) => block,
            };
            Ok(block)
        })
    }
}

impl<'p, T: Real, I: Index> SymmetricBlock<'p, T, I> {
    /// `M = F A_c F` with `F = S R`, formed once with its rows, which decide the clamp; `relax` runs only past the row test, and a frustrated `A_c` takes `S = I` and grounds on its rows at once.
    fn reduce(
        self,
        policy: &ScalingPolicy,
        relax: impl FnOnce(&Self) -> Result<scaling::Relaxation<T>, usize>,
    ) -> Result<Settling<'p, T, I>, BuildError> {
        let at = |i| out_of_range(self.vertices, i);
        let RowSums { diagonal, abs } = &self.entries.rows;
        if let Some(i) = abs.iter().position(|&s| !diagonal_in_range(s)) {
            return Err(at(i));
        }
        let relaxation = match diagonal
            .iter()
            .zip(abs)
            .all(|(&d, &s)| policy.dominates(d, s))
        {
            true => None,
            false => Some(relax(&self).map_err(at)?),
        };
        let folding = folding_signs(&self.entries.off_diagonal);
        let balance = match folding {
            Some(_) => Balance::Balanced,
            None => Balance::Frustrated,
        };
        let signs = folding.unwrap_or(Congruence::Identity);
        let (m, rows, vector) = match relaxation {
            // Signs keep every magnitude, so `A_c`'s rows are `M`'s.
            None => {
                let m = Congruent::signed(self.vertices, self.entries.off_diagonal, signs);
                (m, self.entries.rows, ScaleVector::Identity)
            }
            Some((scales, vector)) => {
                let (m, abs) =
                    Congruent::scaled(self.vertices, self.entries.off_diagonal, signs, &scales)
                        .map_err(at)?;
                let diagonal = (self.entries.rows.diagonal.iter().zip(&scales[..]))
                    .map(|(&a, &f)| f * a * f)
                    .collect();
                (m, RowSums { diagonal, abs }, vector)
            }
        };
        let deficits = (rows.diagonal.iter().zip(&rows.abs)).map(|(&d, &s)| widen(s) - widen(d));
        let clamped = (vector)
            .past_band(
                widen(T::epsilon()),
                deficits,
                rows.diagonal.iter().map(|&d| widen(d)),
                row_degrees(rows.diagonal.len(), m.off_diagonal()),
            )
            .map_or(0.0, |past| past.fold(0.0, f64::max));
        match balance {
            Balance::Balanced => Ok(Settling::Pending(Folded {
                m,
                rows,
                vector,
                clamped,
            })),
            Balance::Frustrated => Ok(Settling::Settled(
                Folded {
                    m,
                    rows,
                    vector,
                    clamped,
                }
                .frustrated()?,
            )),
        }
    }
}

/// A reduced block, its surplus still to be taken from `V`, which only a folded `M` can, or settled.
enum Settling<'p, T, I> {
    Pending(Folded<'p, T, I>),
    Settled(IrreducibleBlock<T, I>),
}

/// A folded `M` with its rows and the bands they sit in, its surplus still to be decided.
struct Folded<'p, T, I> {
    m: Congruent<'p, T, I>,
    rows: RowSums<T>,
    vector: ScaleVector,
    /// The worst `(Σ_j|m_ij| − m_ii)/m_ii` clamped past its row's band.
    clamped: f64,
}

/// `M`'s surplus `M1` and what tells a real one from rounding: a CSR `A` has only `M`'s rows, a Gram also `V`.
trait Surplus<T: Real, I: Index> {
    /// Whether residues alone float a scaled block, which then has no certificate to fall back on.
    const RESIDUES_FLOAT_SCALED: bool;
    /// Each row's residue `(M1)_i`.
    fn residues<'a>(
        &'a self,
        folded: &'a Folded<'_, T, I>,
    ) -> impl Iterator<Item = f64> + Clone + 'a;
    /// The rows whose stated ground passes its band, which no certificate floats; none by default.
    fn stated(&self, _: &Folded<'_, T, I>) -> Option<Box<[bool]>> {
        None
    }
    /// Whether the block's cycles all close within `√ε`; asked last, as it may run a pass over `V`.
    fn certifies(&self, vertices: &[I]) -> bool;
    /// `M`'s grounds once it does not float.
    fn grounds(&self, folded: &Folded<'_, T, I>) -> Grounding<T>;
}

/// Read from `M`'s rows: `d_i − Σ_j |m_ij|`.
struct RowSurplus;

impl<T: Real, I: Index> Surplus<T, I> for RowSurplus {
    const RESIDUES_FLOAT_SCALED: bool = true;

    fn residues<'a>(
        &'a self,
        folded: &'a Folded<'_, T, I>,
    ) -> impl Iterator<Item = f64> + Clone + 'a {
        let RowSums { diagonal, abs } = &folded.rows;
        (diagonal.iter().zip(abs)).map(|(&d, &s)| widen(d) - widen(s))
    }

    fn certifies(&self, _: &[I]) -> bool {
        false
    }

    fn grounds(&self, Folded { m, rows, .. }: &Folded<'_, T, I>) -> Grounding<T> {
        match surplus(rows) {
            grounding if grounding.is_grounded() => grounding,
            // An uncompensated `f64` row sum can round up to `d_i` from degree 67 (#269).
            _ => {
                let sums = m.off_diagonal().compensated_abs_row_sums();
                let residues = (rows.diagonal.iter().zip(sums))
                    .map(|(&d, s)| narrow((widen(d) - s).max(0.0)))
                    .collect();
                Grounding::of(residues)
            }
        }
    }
}

impl<T: Real, I: Index> Folded<'_, T, I> {
    /// The one float-or-ground rule. A row's surplus is rounding within its band, [`ScaleVector::past_band`]: a residue `(M1)_i` at its `k_i` entries, less the block's clamp, rounding too; a stated ground at none, and only where its residue is positive. `M` floats in the singular band with no stated ground and no residue past its band, or on a certificate; grounded, it keeps the rows past their band, or every residue when none is, as a balanced block needs a ground.
    fn settle<S: Surplus<T, I>>(
        self,
        (ones, trace): (f64, f64),
        surplus: &S,
    ) -> Result<IrreducibleBlock<T, I>, BuildError> {
        let stated = surplus.stated(&self);
        let in_band = stated.is_none() && within_band::<T>(ones, trace);
        let residues_float =
            matches!(self.vector, ScaleVector::Identity) || S::RESIDUES_FLOAT_SCALED;
        let certified = || surplus.certifies(self.m.vertices());
        // A scaled Gram floats on its certificate alone, before any residue is read.
        let certificate = in_band && !residues_float && certified();
        let grounds = match certificate {
            true => None,
            false => self.ground_rows(surplus.residues(&self), stated.as_deref()),
        };
        let (vector, clamped) = (self.vector, self.clamped);
        if certificate || (in_band && residues_float && (grounds.is_none() || certified())) {
            return Ok((self.m).block(Balance::Balanced, Grounding::ungrounded(), vector, clamped));
        }
        let grounding = surplus.grounds(&self);
        if !grounding.is_grounded() {
            let past = (grounds.as_deref()).and_then(|grounds| grounds.iter().position(|&g| g));
            let row = past.unwrap_or_else(|| {
                let residues: Vec<f64> = surplus.residues(&self).collect();
                argmax(&residues, |r| r)
            });
            return Err(out_of_range(self.m.vertices(), row));
        }
        let Self { m, .. } = self;
        let grounding = grounding
            .kept(grounds.as_deref())
            .map_err(|i| out_of_range(m.vertices(), i))?;
        Ok(m.block(Balance::Balanced, grounding, vector, clamped))
    }

    /// A frustrated `M`, definite without a ground, keeping none of its rounding.
    fn frustrated(self) -> Result<IrreducibleBlock<T, I>, BuildError> {
        let grounds = self.ground_rows(RowSurplus.residues(&self), None);
        let Self {
            m,
            rows,
            vector,
            clamped,
        } = self;
        let grounding = (grounds.as_deref()).map_or(Ok(Grounding::ungrounded()), |g| {
            surplus(&rows).kept(Some(g))
        });
        let grounding = grounding.map_err(|i| out_of_range(m.vertices(), i))?;
        Ok(m.block(Balance::Frustrated, grounding, vector, clamped))
    }

    /// `(ε·1ᵀM1, ε·tr M)` from `M`'s rows, `1ᵀM1 = tr M − Σ|offdiag(M)|`, so `tr M` cannot overflow.
    fn band(&self) -> (f64, f64) {
        let epsilon = widen(T::epsilon());
        let diagonal = crate::Sum2::of(&self.rows.diagonal, |_, d| epsilon * widen(d));
        let trace = diagonal.total();
        // `tr M` rounded on its own drops a surplus below its ulp, which the Gram path keeps.
        let ones = (self.m.off_diagonal())
            .scaled_abs_into(diagonal, -epsilon)
            .total();
        (ones, trace)
    }

    /// The rows whose surplus is a ground, outweighing its coupling, past its band, or `stated` with a positive residue; `None` when every one is rounding.
    #[inline(always)]
    fn ground_rows(
        &self,
        surplus: impl Iterator<Item = f64> + Clone,
        stated: Option<&[bool]>,
    ) -> Option<Box<[bool]>> {
        let RowSums { diagonal, abs } = &self.rows;
        let stated = |i: usize, s: f64| s > 0.0 && stated.is_some_and(|stated| stated[i]);
        let ground = (surplus.clone().zip(abs).enumerate())
            .map(|(i, (s, &abs))| s > widen(abs) || stated(i, s));
        let excess = (surplus.zip(diagonal)).map(|(s, &d)| s - self.clamped * widen(d));
        let degrees = row_degrees(diagonal.len(), self.m.off_diagonal());
        let diagonal_f64 = diagonal.iter().map(|&d| widen(d));
        let past = (self.vector).past_band(widen(T::epsilon()), excess, diagonal_f64, degrees);
        match past {
            // A floating block's rows: one scan, as the mask is built only once a row holds.
            None if !ground.clone().any(|g| g) => None,
            None => rows_where(diagonal.len(), ground),
            Some(past) => rows_where(
                diagonal.len(),
                ground.zip(past).map(|(g, ratio)| g || ratio > 0.0),
            ),
        }
    }
}

/// The `n` rows where `holds` does, `None` when none: the mask is built only then.
pub(super) fn rows_where(n: usize, holds: impl Iterator<Item = bool>) -> Option<Box<[bool]>> {
    let mut rows: Option<Box<[bool]>> = None;
    for (i, _) in holds.enumerate().filter(|&(_, holds)| holds) {
        rows.get_or_insert_with(|| vec![false; n].into())[i] = true;
    }
    rows
}

/// A residue's terms: at most `n − 1` entries a row, each row's count read on demand.
fn row_degrees<T: Real, I: Index>(
    n: usize,
    off_diagonal: &OffDiagonal<T, I>,
) -> (usize, impl FnOnce() -> Vec<usize> + '_) {
    (n.saturating_sub(1), || off_diagonal.degrees())
}

/// Local row `i` of the block on `vertices` computed a value out of range.
fn out_of_range<I: Index>(vertices: &[I], i: usize) -> BuildError {
    BuildError::OutOfRange {
        vertex: vertices[i].index(),
    }
}

/// Each row's residue, deficits clamped to zero.
fn surplus<T: Real>(RowSums { diagonal, abs }: &RowSums<T>) -> Grounding<T> {
    Grounding::of(
        (diagonal.iter().zip(abs))
            .map(|(&d, &s)| (d - s).max(T::zero()))
            .collect(),
    )
}

/// `1ᵀM1 ≤ SINGULAR_BAND ε tr M` bounds `M`'s least eigenvalue relative to its diagonal (Rayleigh).
const SINGULAR_BAND: f64 = 5.0;

/// Whether `ones = 1ᵀM1` is within the band of `trace = tr M`: needed, not enough, for `M` to float.
pub(crate) fn within_band<T: Real>(ones: f64, trace: f64) -> bool {
    ones <= SINGULAR_BAND * widen(T::epsilon()) * trace
}

/// The signs `S` folding `A_c` into a Z-matrix, `I` when it already is one; `None` when frustrated.
fn folding_signs<T: Real, I: Index>(off_diagonal: &OffDiagonal<T, I>) -> Option<Congruence<T>> {
    if off_diagonal.nonpositive() {
        return Some(Congruence::Identity);
    }
    if let OffDiagonal::Bipartite(cross) = off_diagonal {
        if cross.positive() {
            return Some(Congruence::NegateFrom(cross.n_rows()));
        }
    }
    let mut forest = ParityForest::<I>::new(off_diagonal.n());
    for (i, row) in off_diagonal.upper_rows() {
        for (j, &a) in row {
            // A positive entry folds only across opposite signs.
            forest.join(i, j, a > T::zero())?;
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
        let sign = |negated: bool| if negated { -T::one() } else { T::one() };
        Congruence::Signs(
            (0..self.parent.len())
                .map(|i| sign(self.find(i).1))
                .collect(),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::CsrRef;

    /// A degree-13 star whose centre is 24ε short dominates within the default tolerance: no scaling runs, yet its rows are clamped.
    #[test]
    fn a_deficit_within_the_tolerance_is_clamped_unscaled() {
        let leaves = 13u32;
        let (mut row_ptrs, mut columns) =
            (vec![0u32, leaves + 1], (0..=leaves).collect::<Vec<_>>());
        let mut values = vec![f64::from(leaves) * (1.0 - 24.0 * f64::EPSILON)];
        values.extend(std::iter::repeat_n(-1.0, leaves as usize));
        for leaf in 1..=leaves {
            columns.extend([0, leaf]);
            values.extend([-1.0, 1.0]);
            row_ptrs.push(columns.len() as u32);
        }
        let csr =
            CsrRef::new(leaves as usize + 1, &row_ptrs, &columns, &values).expect("valid shape");
        let policy = ScalingPolicy::default();
        let unscaled =
            |_: &SymmetricBlock<'_, f64, u32>| -> Result<scaling::Relaxation<f64>, usize> {
                panic!("dominant within the tolerance, so nothing scales")
            };
        let partitioned = of_csr::<f64, u32, u32>(Canonical::of(csr.0));
        let summands = partitioned
            .reduced(Threads::All, |vertices, entries, _| match (SymmetricBlock {
                vertices,
                entries,
            })
            .reduce(&policy, unscaled)?
            {
                Settling::Pending(Folded {
                    vector: ScaleVector::Identity,
                    clamped,
                    ..
                }) => Ok(clamped),
                _ => Ok(f64::NAN),
            })
            .expect("admitted");
        let [clamped] = summands.coupled[..] else {
            panic!("one component")
        };
        assert!(clamped > 16.0 * f64::EPSILON, "{clamped:e}");
    }
}
