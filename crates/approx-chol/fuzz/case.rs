use std::error::Error;
use std::num::{NonZeroU32, NonZeroUsize};

use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, TriangleCsrRef,
};
use approx_chol::{factorize_with, ApproxInverse, Config, MemStack, Threads};
use approx_chol::{FrustratedRoute, Real, Sampling, ScalingPolicy, SchurMode};
use serde::{Deserialize, Serialize};

#[path = "cells.rs"]
mod cells;

/// The precision is a variant, so one byte string never decodes as both.
#[derive(Debug, Serialize, Deserialize)]
pub enum Input {
    F64(Case<f64>),
    F32(Case<f32>),
}

impl Input {
    /// The blocks solved, or a typed rejection; a panic is the defect the target hunts.
    pub fn run(&self) -> Result<usize, Box<dyn Error>> {
        match self {
            Self::F64(case) => case.run(),
            Self::F32(case) => case.run(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Case<T> {
    pub view: View<T>,
    pub tolerance: f64,
    // A budget near `usize::MAX` on an uncertifiable block never returns.
    pub max_iterations: u16,
    pub build: Build,
}

/// Every public view; a `u16` `n` only added dead vertices and cut exec/s 50x.
#[derive(Debug, Serialize, Deserialize)]
pub enum View<T> {
    Csr(Rows<T>),
    UpperCsr(Rows<T>),
    Laplacian(Edges<T>),
    Sddm(Edges<T>, Vec<T>),
    FactorWidth2 {
        n: u8,
        a: Vec<u32>,
        b: Vec<u32>,
        x: Vec<T>,
        y: Vec<T>,
    },
    BipartiteFactorWidth2 {
        left: Side<T>,
        right: Side<T>,
        scale: Option<Vec<T>>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Edges<T> {
    pub n: u8,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub weights: Option<Vec<T>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Rows<T> {
    pub n: u8,
    pub row_ptrs: Vec<u32>,
    pub columns: Vec<u32>,
    pub values: Vec<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Side<T> {
    pub n: u8,
    pub rows: Vec<u32>,
    pub values: Option<Vec<T>>,
}

/// `Config` in a `u8` per count knob: `None` is an exact complement, under two copies sample once, a `u8` caps AC2; the frustrated route as itself.
#[derive(Debug, Serialize, Deserialize)]
pub struct Build {
    pub seed: u64,
    pub complement: Option<u8>,
    pub approximate: u8,
    pub exact_max_dim: u8,
    pub frustrated: FrustratedRoute,
}

impl Build {
    pub fn config(&self) -> Config {
        let complement = (self.complement).map_or(SchurMode::Exact, |k| SchurMode::Approximate {
            sampling: clique_tree(k),
        });
        Config {
            seed: self.seed,
            sampling: clique_tree(self.approximate),
            frustrated: self.frustrated,
            exact_max_dim: self.exact_max_dim.into(),
            complement,
        }
    }
}

/// Every byte a count: `0` samples as `1`, standard AC.
fn clique_tree(k: u8) -> Sampling {
    Sampling::CliqueTree {
        copies: NonZeroU32::new(k.into()).unwrap_or(NonZeroU32::MIN),
    }
}

impl<T: Real> Edges<T> {
    fn laplacian(&self) -> Result<Laplacian<'_, T>, Box<dyn Error>> {
        let n = self.n.into();
        Ok(match &self.weights {
            Some(w) => Laplacian::weighted(n, &self.a, &self.b, w)?,
            None => Laplacian::new(n, &self.a, &self.b)?,
        })
    }
}

impl<T: Real> Side<T> {
    fn factor(&self) -> Result<FactorWidth1<'_, T>, Box<dyn Error>> {
        let n = self.n.into();
        Ok(match &self.values {
            Some(values) => FactorWidth1::valued(n, &self.rows, values)?,
            None => FactorWidth1::new(n, &self.rows)?,
        })
    }
}

impl<T: Real> Case<T> {
    pub fn factor(&self) -> Result<ApproxInverse<T>, Box<dyn Error>> {
        let policy = ScalingPolicy::new(self.tolerance, self.max_iterations.into())
            .ok_or("tolerance outside [0, 1)")?;
        let config = self.build.config();
        Ok(match &self.view {
            View::Csr(r) => {
                let csr = CsrRef::new(r.n.into(), &r.row_ptrs, &r.columns, &r.values)?;
                factorize_with(csr, config, policy)?
            }
            View::UpperCsr(r) => {
                let csr = TriangleCsrRef::upper(r.n.into(), &r.row_ptrs, &r.columns, &r.values)?;
                factorize_with(csr, config, policy)?
            }
            View::Laplacian(edges) => factorize_with(edges.laplacian()?, config, policy)?,
            View::Sddm(edges, surplus) => {
                factorize_with(Sddm::new(edges.laplacian()?, surplus)?, config, policy)?
            }
            View::FactorWidth2 { n, a, b, x, y } => {
                let gram = FactorWidth2::new((*n).into(), (a, x), (b, y))?;
                factorize_with(gram, config, policy)?
            }
            View::BipartiteFactorWidth2 { left, right, scale } => {
                let (left, right) = (left.factor()?, right.factor()?);
                let gram = match scale {
                    Some(s) => BipartiteFactorWidth2::scaled(left, right, s)?,
                    None => BipartiteFactorWidth2::new(left, right)?,
                };
                factorize_with(gram, config, policy)?
            }
        })
    }

    fn run(&self) -> Result<usize, Box<dyn Error>> {
        let factor = self.factor()?;
        let b = vec![T::one(); factor.n()];
        let mut solved = b.clone();
        factor
            .apply(&mut solved, Threads::All)
            .expect("a right-hand side of length n is accepted");
        // NaN scratch reaches the solution if any slot is read before written.
        for threads in [Threads::All, Threads::Count(NonZeroUsize::MIN)] {
            let mut x = b.clone();
            let mut cells = cells::filled(factor.scratch(1, threads), T::nan());
            factor
                .apply_with(&mut x, threads, MemStack::new_any(&mut cells))
                .expect("a stack of the factor's own scratch is accepted");
            let agrees =
                (solved.iter().zip(&x)).all(|(s, x)| s.integer_decode() == x.integer_decode());
            assert!(agrees, "dirty scratch changed the solution");
        }
        let components = factor.components().len();
        Ok(components)
    }
}
