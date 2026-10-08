mod csr;
mod sets;

use crate::graph::{BlockLayout, BlockVertices, Component};
use crate::types::Real;
use crate::{AdjacencyError, LaplacianError, SddmError, SurplusDefect, WeightDefect};
use sets::DisjointSets;

/// An admission threshold, measured to keep uniformly scaled solves at unit-scale quality (#163).
fn floor<T: Real>() -> T {
    T::min_positive_value() / T::epsilon()
}

/// The one rule every stored edge weight passes, whichever route stores it.
#[inline]
fn check_weight<T: Real>(weight: T) -> Result<T, WeightDefect> {
    if !weight.is_finite() {
        Err(WeightDefect::NonFinite)
    } else if weight <= T::zero() {
        Err(WeightDefect::NotPositive)
    } else if weight < floor() {
        Err(WeightDefect::BelowFloor)
    } else {
        Ok(weight)
    }
}

/// A graph Laplacian `L(G)`, stored as `G`'s strict upper adjacency and verified when built.
#[derive(Clone, Debug)]
pub struct Laplacian<T = f64> {
    row_ptrs: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<T>,
    components: Components,
}

/// How the vertices fall into connected components, found while the edges were checked.
#[derive(Clone, Debug)]
pub(crate) enum Components {
    /// Exactly one component.
    Connected,
    /// Any other count, so `n = 0` is zero blocks.
    Split(BlockLayout),
}

impl<T> Laplacian<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    /// Row `i`'s neighbors `j > i` ascending in `neighbors[row_ptrs[i]..row_ptrs[i + 1]]`, each with its weight.
    pub fn new(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
    ) -> Result<Self, LaplacianError> {
        let components = traverse(&row_ptrs, &neighbors, &weights)?
            .finish(None)
            .map_err(|NotFinite { vertex }| LaplacianError::DegreeNotFinite { vertex })?;
        Ok(Self {
            row_ptrs,
            neighbors,
            weights,
            components,
        })
    }
}

impl<T> Laplacian<T> {
    /// The number of vertices.
    pub fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    #[inline]
    pub(crate) fn row(&self, i: usize) -> (&[u32], &[T]) {
        let (from, to) = (self.row_ptrs[i] as usize, self.row_ptrs[i + 1] as usize);
        (&self.neighbors[from..to], &self.weights[from..to])
    }
}

/// A symmetric diagonally dominant matrix with non-positive off-diagonals: a [`Laplacian`] plus a diagonal surplus.
#[derive(Clone, Debug)]
pub struct Sddm<T = f64> {
    laplacian: Laplacian<T>,
    /// `Some` exactly when some component's surplus total is positive.
    surplus: Option<Vec<T>>,
}

impl<T> From<Laplacian<T>> for Sddm<T> {
    fn from(laplacian: Laplacian<T>) -> Self {
        Self {
            laplacian,
            surplus: None,
        }
    }
}

impl<T> Sddm<T> {
    /// The number of vertices.
    pub fn n(&self) -> usize {
        self.laplacian.n()
    }
}

impl<T> Sddm<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    /// [`Laplacian::new`]'s arrays plus each vertex's diagonal excess over its weighted degree.
    pub fn new(
        row_ptrs: Vec<u32>,
        neighbors: Vec<u32>,
        weights: Vec<T>,
        surplus: Vec<T>,
    ) -> Result<Self, SddmError> {
        let incidence = traverse(&row_ptrs, &neighbors, &weights)?;
        check_surplus(&surplus, row_ptrs.len() - 1)?;
        let components = incidence
            .finish(Some(&surplus))
            .map_err(|NotFinite { vertex }| SddmError::DiagonalNotFinite { vertex })?;
        let surplus = grounding(&components, surplus)
            .map_err(|GroundOverflow { vertex }| SddmError::GroundOverflow { vertex })?;
        let laplacian = Laplacian {
            row_ptrs,
            neighbors,
            weights,
            components,
        };
        Ok(Self { laplacian, surplus })
    }

    /// Each component mapped in layout order, and the order itself, which is the factor's permutation.
    pub(crate) fn map_components<B, E>(
        self,
        mut each: impl FnMut(&Component<'_, T>) -> Result<B, E>,
    ) -> Result<(Vec<B>, Option<Vec<u32>>), E> {
        let surplus = self.surplus.as_deref();
        let mapped = match &self.laplacian.components {
            Components::Connected => {
                let whole = BlockVertices::whole(self.laplacian.n());
                vec![each(&Component::new(&self.laplacian, whole, surplus))?]
            }
            Components::Split(layout) => {
                let mut mapped = Vec::with_capacity(layout.block_count());
                // Scratch reused across blocks; each view refills the entries it names.
                let mut local_of = vec![0u32; self.laplacian.n()];
                for vertices in layout.blocks() {
                    let part = BlockVertices::part(vertices, &mut local_of);
                    mapped.push(each(&Component::new(&self.laplacian, part, surplus))?);
                }
                mapped
            }
        };
        let order = match self.laplacian.components {
            Components::Connected => None,
            Components::Split(layout) => Some(layout.into_order()),
        };
        Ok((mapped, order))
    }
}

/// The caller's arrays checked in one walk, each edge fed to the incidence as it passes.
fn traverse<T: Real>(
    row_ptrs: &[u32],
    neighbors: &[u32],
    weights: &[T],
) -> Result<Incidence<T>, AdjacencyError> {
    let (&first, &end) = row_ptrs
        .first()
        .zip(row_ptrs.last())
        .ok_or(AdjacencyError::RowPtrsEmpty)?;
    if first != 0 {
        return Err(AdjacencyError::RowPtrsMustStartAtZero { got: first });
    }
    if neighbors.len() != weights.len() {
        return Err(AdjacencyError::NeighborsWeightsLenMismatch {
            neighbors: neighbors.len(),
            weights: weights.len(),
        });
    }
    if end as usize != neighbors.len() {
        return Err(AdjacencyError::RowPtrsEndMismatch {
            end,
            len: neighbors.len(),
        });
    }
    // Checked first, so every row's range lies inside the arrays.
    if let Some(row) = row_ptrs.windows(2).position(|pair| pair[0] > pair[1]) {
        return Err(AdjacencyError::RowPtrsDecrease { row });
    }
    let n = row_ptrs.len() - 1;
    // A grounded component's ground slot is named by a `u32` after its vertices.
    if u32::try_from(n + 1).is_err() {
        return Err(AdjacencyError::TooManyVertices { n });
    }
    let mut incidence = Incidence::new(n);
    for row in 0..n {
        let mut links = incidence.row(row);
        let mut previous = row;
        for at in row_ptrs[row] as usize..row_ptrs[row + 1] as usize {
            let col = neighbors[at] as usize;
            let edge = (row, col);
            if col <= previous {
                return Err(if col <= row {
                    AdjacencyError::NotStrictlyUpper { edge }
                } else {
                    AdjacencyError::Unsorted { edge }
                });
            }
            if col >= n {
                return Err(AdjacencyError::NeighborOutOfBounds { edge, n });
            }
            let weight = check_weight(weights[at])
                .map_err(|defect| AdjacencyError::Weight { edge, defect })?;
            links.add(col, weight);
            previous = col;
        }
    }
    Ok(incidence)
}

/// Not finiteness: a non-finite surplus makes its diagonal non-finite, which [`Incidence::finish`] reports.
fn check_surplus<T: Real>(surplus: &[T], n: usize) -> Result<(), SddmError> {
    if surplus.len() != n {
        return Err(SddmError::SurplusLength {
            len: surplus.len(),
            n,
        });
    }
    for (vertex, &s) in surplus.iter().enumerate() {
        let defect = if s < T::zero() {
            SurplusDefect::Negative
        } else if s > T::zero() && s < floor() {
            SurplusDefect::BelowFloor
        } else {
            continue;
        };
        return Err(SddmError::Surplus { vertex, defect });
    }
    Ok(())
}

/// Each vertex's weighted degree and component, accumulated over the same edges.
struct Incidence<T> {
    degrees: Vec<T>,
    sets: DisjointSets,
}

/// A vertex whose diagonal, its degree plus any surplus, is not finite.
struct NotFinite {
    vertex: usize,
}

impl<T: Real> Incidence<T> {
    fn new(n: usize) -> Self {
        Self {
            degrees: vec![T::zero(); n],
            sets: DisjointSets::new(n),
        }
    }

    /// Resolves the row's set once, so each of its edges unions from a known root.
    #[inline]
    fn row(&mut self, row: usize) -> RowIncidence<'_, T> {
        let root = self.sets.find(row as u32);
        RowIncidence {
            incidence: self,
            row,
            root,
        }
    }

    /// Summed in storage order, the order every diagonal is summed in; the degrees never leave.
    fn finish(self, surplus: Option<&[T]>) -> Result<Components, NotFinite> {
        let first = match surplus {
            None => self.degrees.iter().position(|degree| !degree.is_finite()),
            Some(surplus) => self
                .degrees
                .iter()
                .zip(surplus)
                .position(|(&degree, &s)| !(degree + s).is_finite()),
        };
        if let Some(vertex) = first {
            return Err(NotFinite { vertex });
        }
        Ok(self.sets.components())
    }
}

struct RowIncidence<'a, T> {
    incidence: &'a mut Incidence<T>,
    row: usize,
    root: u32,
}

impl<T: Real> RowIncidence<'_, T> {
    #[inline]
    fn add(&mut self, col: usize, weight: T) {
        let degrees = &mut self.incidence.degrees;
        degrees[self.row] = degrees[self.row] + weight;
        degrees[col] = degrees[col] + weight;
        self.root = self.incidence.sets.union_resolved(self.root, col as u32);
    }
}

/// A component whose ground, the sum of its surplus, is not finite.
struct GroundOverflow {
    vertex: usize,
}

/// `None` when no component holds surplus, so a floating input never carries a zero surplus.
fn grounding<T: Real>(
    components: &Components,
    surplus: Vec<T>,
) -> Result<Option<Vec<T>>, GroundOverflow> {
    let grounded = match components {
        Components::Connected => grounds(0, &surplus, 0..surplus.len())?,
        Components::Split(layout) => layout.blocks().try_fold(false, |any, vertices| {
            let members = vertices.iter().map(|&v| v as usize);
            Ok(grounds(vertices[0] as usize, &surplus, members)? | any)
        })?,
    };
    Ok(grounded.then_some(surplus))
}

/// Sums the ground's degree, which the approximate arm sums again when it eliminates the ground.
fn grounds<T: Real>(
    first: usize,
    surplus: &[T],
    members: impl Iterator<Item = usize>,
) -> Result<bool, GroundOverflow> {
    let total = members.fold(T::zero(), |sum, v| sum + surplus[v]);
    if !total.is_finite() {
        return Err(GroundOverflow { vertex: first });
    }
    Ok(total > T::zero())
}
