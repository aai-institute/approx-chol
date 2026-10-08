//! Prototype: factor-width-2 inputs `A = V W Vᵀ`, one part per connected component.
//!
//! Each part is scaled, folded or covered, and grounded from `V`; [`Sgdd::factorize`]
//! batch-eliminates its larger side and factors the Schur complement with the kernel.

pub mod bipartite;
mod convert;
mod cross;
mod factor;
mod forest;
mod scaling;
mod schur;

pub use factor::SgddFactor;

use bipartite::{Gram, Graph};
use convert::{Class, Oriented, Uncertified};
use cross::Cross;
use forest::ScaledForest;
use num_traits::PrimInt;

/// Certification policy for a dominance scaling; a deficit left at hand-over is clamped and reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingPolicy {
    tolerance: f64,
    max_iterations: usize,
}

impl ScalingPolicy {
    /// `None` unless `tolerance` is finite and non-negative.
    pub fn new(tolerance: f64, max_iterations: usize) -> Option<Self> {
        (tolerance.is_finite() && tolerance >= 0.0).then_some(Self {
            tolerance,
            max_iterations,
        })
    }
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            tolerance: 1e-9,
            max_iterations: 2048,
        }
    }
}

/// The worst uncertified scaling across parts; present iff some violation exceeded the tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingReport {
    /// Scaling iterations spent on the worst part.
    pub iterations: usize,
    /// Largest relative dominance violation, clamped away.
    pub violation: f64,
}

/// Why a view could not become an [`Sgdd`]; `vertex` is the part's first vertex in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SgddError {
    /// A scaling or a scaled entry left the finite range.
    Unscalable {
        /// First vertex of the offending part.
        vertex: u32,
    },
}

impl core::fmt::Display for SgddError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cannot convert to Sgdd: {self:?}")
    }
}

impl std::error::Error for SgddError {}

/// How the larger side is eliminated before the kernel factors the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchurMode {
    /// Clique-tree sampling per eliminated star, as within's sampled Schur.
    Sampled {
        /// Star streams derive from this seed by star index.
        seed: u64,
        /// Star-edge split factor; `1` is standard.
        split: u32,
    },
    /// The exact Schur complement.
    Exact,
}

impl Default for SchurMode {
    fn default() -> Self {
        Self::Sampled { seed: 0, split: 1 }
    }
}

pub(crate) struct NotScalable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gauge {
    Floating,
    Grounded,
}

/// Whether the part's core is the folded matrix or its Gremban cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    Folded,
    Cover,
}

/// The diagonal map from `A`'s coordinates to the core's: `x = F y`, `F` an involution up to scale.
#[derive(Clone, Debug)]
pub(crate) enum Coordinates {
    /// `+1` on the eliminated side, `-1` on the kept one.
    Canonical,
    /// Sign times dominance scale per local vertex.
    Scaled(Vec<f64>),
}

/// One component: `[D_e, −C; −Cᵀ, D_k]` in core coordinates, eliminated (larger) side first.
#[derive(Clone, Debug)]
pub(crate) struct Part {
    pub(crate) cross: Cross,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) surplus: Vec<f64>,
    pub(crate) globals: Vec<u32>,
    pub(crate) gauge: Gauge,
    pub(crate) form: Form,
    pub(crate) coordinates: Coordinates,
}

/// Symmetric generalized diagonally dominant matrix as one certified part per component.
#[derive(Clone, Debug)]
pub struct Sgdd {
    n: usize,
    parts: Vec<Part>,
    report: Option<ScalingReport>,
}

impl Sgdd {
    /// Dimension of `A`: left vertices, then right ones.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Present iff some part's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.report
    }

    /// Scales, folds or covers each component, and grounds it from the Gram's columns.
    pub fn from_gram<I: PrimInt>(
        gram: Gram<'_, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, SgddError> {
        let (n_left, n_right) = gram.sizes();
        let (cross, diagonal) = cross::of_gram(&gram);
        let mut forest = ScaledForest::grow(&gram, &cross.c);
        let components = forest.components();
        let single = components.len() == 1;
        let mut remap = vec![u32::MAX; n_right];
        let mut levels: Vec<Option<(usize, f64)>> = vec![None; n_left + n_right];
        let mut parts: Vec<(Part, Gauge)> = Vec::with_capacity(components.len());
        let mut report: Option<ScalingReport> = None;
        for (root, members) in &components {
            let Some(oriented) = Oriented::new(members, &cross, &diagonal, &mut remap, single)
            else {
                continue;
            };
            let swapped = oriented.swapped;
            let vertex = oriented.globals[0];
            let forest_gauge = forest.gauge(*root);
            let (part, uncertified) = oriented
                .convert(Class::General, &policy)
                .map_err(|NotScalable| SgddError::Unscalable { vertex })?;
            if let Some(Uncertified {
                iterations,
                violation,
            }) = uncertified
            {
                if report.is_none_or(|worst| violation > worst.violation) {
                    report = Some(ScalingReport {
                        iterations,
                        violation,
                    });
                }
            }
            if part.form == Form::Folded {
                let n_eliminated = part.cross.n_rows();
                let (row_offset, col_offset) = if swapped {
                    (n_eliminated, 0)
                } else {
                    (0, n_eliminated)
                };
                for (p, &i) in members.rows.iter().enumerate() {
                    levels[i] = Some((parts.len(), part.factor(row_offset + p)));
                }
                for (q, &j) in members.cols.iter().enumerate() {
                    levels[n_left + j] = Some((parts.len(), part.factor(col_offset + q)));
                }
            }
            parts.push((part, forest_gauge));
        }
        let scaled = forest::scaled_gauges(&gram, &levels, n_left, parts.len());
        let parts = (parts.into_iter().enumerate())
            .map(|(k, (mut part, forest_gauge))| {
                let gauge = match part.form {
                    Form::Folded => scaled[k],
                    Form::Cover => forest_gauge,
                };
                if gauge == Gauge::Floating {
                    part.float();
                }
                part
            })
            .collect();
        Ok(Self {
            n: n_left + n_right,
            parts,
            report,
        })
    }
}

impl<'a, I: PrimInt> TryFrom<Graph<'a, I>> for Sgdd {
    type Error = SgddError;

    /// Bipartite, balanced and of zero surplus by shape: every part floats, `D = I`.
    fn try_from(graph: Graph<'a, I>) -> Result<Self, Self::Error> {
        let (n_left, n_right) = graph.sizes();
        let cross = cross::of_graph(&graph);
        let diagonal: Vec<f64> = (0..cross.n_local())
            .map(|i| cross.neighbors(i).map(|(_, v)| v).sum())
            .collect();
        let components = forest::graph_components(&cross);
        let single = components.len() == 1;
        let mut remap = vec![u32::MAX; n_right];
        let mut parts = Vec::with_capacity(components.len());
        for members in &components {
            let Some(oriented) = Oriented::new(members, &cross, &diagonal, &mut remap, single)
            else {
                continue;
            };
            let vertex = oriented.globals[0];
            let (part, _) = oriented
                .convert(Class::Known, &ScalingPolicy::default())
                .map_err(|NotScalable| SgddError::Unscalable { vertex })?;
            parts.push(part);
        }
        Ok(Self {
            n: n_left + n_right,
            parts,
            report: None,
        })
    }
}

/// Neumaier compensated sum: a flat sum biases a large part's mean.
pub(crate) fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for &value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - next) + value;
        } else {
            compensation += (value - next) + sum;
        }
        sum = next;
    }
    sum + compensation
}
