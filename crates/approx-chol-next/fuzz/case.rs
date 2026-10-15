use std::error::Error;

use approx_chol_next::input::{
    BipartiteGram, Csr, FactorWidth1, FactorWidth2, Laplacian, SignedGraph, UpperCsr, Weights,
};
use approx_chol_next::{factorize, CliqueRule, Config, Factor, Index, InnerParallelism};
use approx_chol_next::{Multiplicity, Real, ScalingPolicy, Sgdd, SplitCount};
use serde::{Deserialize, Serialize};

/// The precision is a variant, so one byte string never decodes as both.
#[derive(Debug, Serialize, Deserialize)]
pub enum Input {
    F64(Case<f64>),
    F32(Case<f32>),
}

impl Input {
    /// `Err` is a typed rejection; a panic is the defect the target hunts.
    pub fn run(&self) -> Result<(), Box<dyn Error>> {
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
    Gram {
        n: u8,
        a: Vec<u32>,
        b: Vec<u32>,
        x: Vec<T>,
        y: Vec<T>,
    },
    BipartiteGram {
        left: Side<T>,
        right: Side<T>,
        weights: Weighting<T>,
    },
    Laplacian {
        n: u8,
        a: Vec<u32>,
        b: Vec<u32>,
        weights: Weighting<T>,
        ground: Option<Vec<T>>,
    },
    SignedGraph {
        n: u8,
        a: Vec<u32>,
        b: Vec<u32>,
        positive: Vec<bool>,
        weights: Weighting<T>,
    },
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

#[derive(Debug, Serialize, Deserialize)]
pub enum Weighting<T> {
    Unit,
    Of(Vec<T>),
    SqrtOf(Vec<T>),
}

/// `Config` without serde; `copies` below two samples once, and a `u8` caps AC2's fill.
#[derive(Debug, Serialize, Deserialize)]
pub struct Build {
    pub seed: u64,
    pub exact_complement: bool,
    pub copies: u8,
    pub dense_up_to: u8,
}

impl Build {
    pub fn config(&self) -> Config {
        let copies =
            SplitCount::new(self.copies.into()).map_or(Multiplicity::Single, Multiplicity::Split);
        let complement = match self.exact_complement {
            true => CliqueRule::Exact,
            false => CliqueRule::Sampled(copies),
        };
        Config {
            seed: self.seed,
            complement,
            approximate: copies,
            dense_up_to: self.dense_up_to.into(),
        }
    }
}

impl<T: Real> Weighting<T> {
    fn weights(&self) -> Result<Weights<'_, T>, Box<dyn Error>> {
        Ok(match self {
            Self::Unit => Weights::unit(),
            Self::Of(w) => Weights::of(w)?,
            Self::SqrtOf(s) => Weights::sqrt_of(s)?,
        })
    }
}

impl<T: Real> Side<T> {
    fn factor(&self) -> Result<FactorWidth1<'_, T>, Box<dyn Error>> {
        Ok(FactorWidth1::new(
            self.n.into(),
            &self.rows,
            self.values.as_deref(),
        )?)
    }
}

impl<T: Real> Case<T> {
    pub fn factor(&self) -> Result<Factor<T>, Box<dyn Error>> {
        Ok(factorize(&self.reduce()?, &self.build.config()))
    }

    fn run(&self) -> Result<(), Box<dyn Error>> {
        let factor = self.factor()?;
        let b = vec![T::one(); factor.n()];
        let solved = factor
            .solve(&b)
            .expect("a right-hand side of length n is accepted");
        // NaN scratch reaches the solution if any slot is read before written.
        for block in factor.into_blocks() {
            let mut local = vec![T::one(); block.vertices().len()];
            let mut scratch = vec![T::nan(); block.scratch_len()];
            block
                .solve_local(&mut local, &mut scratch, InnerParallelism::Allow)
                .expect("buffers of the block's own lengths are accepted");
            let agrees = (block.vertices().iter().zip(&local))
                .all(|(v, x)| solved[v.index()].integer_decode() == x.integer_decode());
            assert!(agrees, "dirty scratch changed the solution");
        }
        Ok(())
    }

    fn reduce(&self) -> Result<Sgdd<T>, Box<dyn Error>> {
        let policy = ScalingPolicy::new(self.tolerance, self.max_iterations.into())
            .ok_or("tolerance outside [0, 1)")?;
        Ok(match &self.view {
            View::Csr(r) => Sgdd::from_csr(
                Csr::new(r.n.into(), &r.row_ptrs, &r.columns, &r.values)?,
                policy,
            )?,
            View::UpperCsr(r) => {
                let csr = UpperCsr::new(r.n.into(), &r.row_ptrs, &r.columns, &r.values)?;
                Sgdd::from_upper_csr(csr, policy)?
            }
            View::Gram { n, a, b, x, y } => {
                Sgdd::from_gram(FactorWidth2::new((*n).into(), a, b, x, y)?, policy)?
            }
            View::BipartiteGram {
                left,
                right,
                weights,
            } => {
                let gram = BipartiteGram::new(left.factor()?, right.factor()?, weights.weights()?)?;
                Sgdd::from_bipartite_gram(gram, policy)?
            }
            View::Laplacian {
                n,
                a,
                b,
                weights,
                ground,
            } => {
                let graph = Laplacian::new((*n).into(), a, b, weights.weights()?)?;
                let graph = match ground {
                    Some(g) => graph.grounded(g)?,
                    None => graph,
                };
                Sgdd::try_from(graph)?
            }
            View::SignedGraph {
                n,
                a,
                b,
                positive,
                weights,
            } => Sgdd::try_from(SignedGraph::new(
                (*n).into(),
                a,
                b,
                positive,
                weights.weights()?,
            )?)?,
        })
    }
}
