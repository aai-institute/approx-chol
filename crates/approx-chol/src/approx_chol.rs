mod builder;
pub(crate) mod clique_tree;
pub(crate) mod decomposition;
mod star;

pub use builder::Builder;
pub use decomposition::{Deficit, Factor, SolveError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for approximate Cholesky factorization.
///
/// Use [`Default`] for standard AC (recommended for most inputs).
/// Set [`split_merge`](Self::split_merge) to enable AC2.
///
/// # Examples
///
/// ```
/// use approx_chol::Config;
///
/// // AC2 variant with multi-edge multiplicity
/// let config = Config {
///     split_merge: Some(2),
///     seed: 42,
///     ..Default::default()
/// };
/// assert!(config.split_merge.is_some());
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    /// Random seed for the edge-weight sampler. Use different values to get
    /// reproducible but varied approximate factors.
    pub seed: u64,
    /// AC2 multi-edge multiplicity parameter (`k`).
    ///
    /// `None` = standard AC (default), `Some(k)` = AC2 with `k` edge copies
    /// and `k` merge cap per neighbor pair.
    pub split_merge: Option<u32>,
    /// What the caller asserts about the matrix. [`InputClass::Auto`] detects
    /// and handles sign structure and dominance.
    #[cfg_attr(feature = "serde", serde(default))]
    pub assume: InputClass,
    /// Scaling relaxation budget/tolerance and residual-deficit reaction.
    #[cfg_attr(feature = "serde", serde(default))]
    pub scaling: Scaling,
}

/// What the caller asserts about the input matrix. A specific class skips the
/// matching detection pass and turns a violated assertion into an error.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputClass {
    /// Detect sign structure and dominance; fold and scale as needed. The entry
    /// point for general (signed, non-dominant) H-matrices — a specific class
    /// only asserts narrower structure to skip a detection pass.
    #[default]
    Auto,
    /// Graph Laplacian: sign-free, zero row sums.
    Laplacian,
    /// Symmetric diagonally dominant M-matrix: sign-free, dominant.
    Sddm,
    /// Symmetric diagonally dominant: possibly signed, dominant.
    Sdd,
}

/// Scaling relaxation toward generalized diagonal dominance.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaling {
    /// Maximum relaxation sweeps (divergence guard).
    pub budget: u32,
    /// Acceptance slack for weak diagonal dominance.
    pub slack: f64,
    /// Reaction when a residual dominance deficit remains after scaling.
    pub on_deficit: OnDeficit,
}

impl Default for Scaling {
    fn default() -> Self {
        Self {
            budget: 100,
            slack: 1e-6,
            on_deficit: OnDeficit::Error,
        }
    }
}

/// Reaction when scaling leaves a residual deficit (clamped either way).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnDeficit {
    /// Return an error carrying the residual deficit.
    #[default]
    Error,
    /// Proceed with the clamped factor; report the deficit as a diagnostic.
    Warn,
}
