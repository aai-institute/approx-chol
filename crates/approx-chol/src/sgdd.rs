//! Prototype: factor-width-2 inputs `A = V W Vᵀ`, one part per connected component.
//!
//! within's pair pipeline, transplanted: each part is scaled, folded or covered, and grounded
//! from `V`; [`Sgdd::factorize`] batch-eliminates its larger side and factors the Schur complement.

pub mod bipartite;
mod block_elim;
mod cross_tab;
mod csr_block;
mod factor_pairs;
#[doc(hidden)]
pub mod probe;

use bipartite::{Gram, Graph};
use block_elim::{BlockElimSolver, LocalSolverConfig};
use factor_pairs::{cross_tab, split_into_subdomains, LocalDomain};
use num_traits::PrimInt;
use rayon::prelude::*;

use crate::{Backend, Config, Error, Fallback, SolveError};

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

/// Symmetric generalized diagonally dominant matrix as one certified part per component.
#[derive(Clone, Debug)]
pub struct Sgdd {
    n: usize,
    parts: Vec<LocalDomain>,
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
        gram: Gram<'_, '_, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, SgddError> {
        let (n_left, n_right) = gram.sizes();
        let full_ct = cross_tab(gram.columns(), n_left, n_right);
        let (parts, report) =
            split_into_subdomains(gram.columns(), full_ct, Some(gram.diagonals()), &policy)?;
        Ok(Self {
            n: n_left + n_right,
            parts,
            report,
        })
    }

    /// Factors every part, in parallel; `config` routes each reduced system as [`Builder`](crate::low_level::Builder) routes a block.
    pub fn factorize(self, config: &Config, mode: SchurMode) -> Result<SgddFactor, Error> {
        let n = self.n;
        let parts: Vec<SgddPart> = (self.into_components().into_par_iter())
            .map(|component| component.factorize(config, mode))
            .collect::<Result<_, Error>>()?;
        let fallbacks = (parts.iter())
            .flat_map(|part| part.solver.fallbacks().iter().copied())
            .collect();
        Ok(SgddFactor {
            n,
            parts,
            fallbacks,
        })
    }

    /// The converted parts, each factored on its own.
    #[doc(hidden)]
    pub fn into_components(self) -> Vec<SgddComponent> {
        self.parts.into_iter().map(SgddComponent).collect()
    }
}

/// One converted component, not yet factored.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SgddComponent(LocalDomain);

impl SgddComponent {
    /// The component's vertices in `A`'s numbering, in local order.
    pub fn globals(&self) -> &[u32] {
        &self.0.globals
    }

    /// Eliminates the larger side and factors the rest, as [`Sgdd::factorize`] does per part.
    pub fn factorize(self, config: &Config, mode: SchurMode) -> Result<SgddPart, Error> {
        let local = LocalSolverConfig {
            approx_chol: *config,
            schur: mode,
            dense_threshold: match config.backend {
                Backend::ExactBelow { max_dim, .. } => max_dim,
                Backend::Approximate => 0,
            },
        };
        let LocalDomain { globals, component } = self.0;
        let solver = BlockElimSolver::build(component, &local)?;
        Ok(SgddPart { globals, solver })
    }
}

impl<'a, I: PrimInt> TryFrom<Graph<'a, I>> for Sgdd {
    type Error = SgddError;

    /// Bipartite, balanced and of zero surplus by shape: every part floats.
    fn try_from(graph: Graph<'a, I>) -> Result<Self, Self::Error> {
        let (n_left, n_right) = graph.sizes();
        let full_ct = cross_tab(graph.columns(), n_left, n_right);
        let (parts, _) =
            split_into_subdomains(graph.columns(), full_ct, None, &ScalingPolicy::default())?;
        Ok(Self {
            n: n_left + n_right,
            parts,
            report: None,
        })
    }
}

/// Applies each part's solver in `A`'s coordinates; vertices in no part solve to `0`.
#[derive(Clone, Debug)]
pub struct SgddFactor {
    n: usize,
    parts: Vec<SgddPart>,
    fallbacks: Vec<Fallback>,
}

impl SgddFactor {
    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Reduced systems the dense arm declined, factored approximately instead.
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
    }

    /// Solves part by part; panics unless `b` has length [`n`](Self::n).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SolveError> {
        assert_eq!(b.len(), self.n, "rhs length must equal n");
        let mut x = vec![0.0; self.n];
        let (mut rhs, mut sol) = (Vec::new(), Vec::new());
        for part in &self.parts {
            let size = part.scratch_size();
            rhs.clear();
            rhs.resize(size, 0.0);
            sol.clear();
            sol.resize(size, 0.0);
            for (slot, &g) in rhs.iter_mut().zip(&part.globals) {
                *slot = b[g as usize];
            }
            part.solve_local(&mut rhs, &mut sol, true)?;
            for (&g, &value) in part.globals.iter().zip(&sol) {
                x[g as usize] = value;
            }
        }
        Ok(x)
    }

    /// The parts, each a Schwarz-ready local solver.
    #[doc(hidden)]
    pub fn into_parts(self) -> Vec<SgddPart> {
        self.parts
    }
}

/// One component's local solver over its vertices of `A`.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SgddPart {
    globals: Vec<u32>,
    solver: BlockElimSolver,
}

impl SgddPart {
    /// The part's vertices in `A`'s numbering, in local order.
    pub fn globals(&self) -> &[u32] {
        &self.globals
    }

    /// Local dimension.
    pub fn n_local(&self) -> usize {
        self.solver.n_local()
    }

    /// Length of both buffers [`solve_local`](Self::solve_local) takes.
    pub fn scratch_size(&self) -> usize {
        self.solver.scratch_size()
    }

    /// Work a nested parallel solve could split; `0` when there is none.
    pub fn inner_parallelism_work_estimate(&self) -> usize {
        self.solver.inner_parallelism_work_estimate()
    }

    /// Solves into `sol`; `rhs` is overwritten, both are [`scratch_size`](Self::scratch_size) long.
    pub fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        self.solver.solve_local(rhs, sol, allow_inner_parallelism)
    }
}
