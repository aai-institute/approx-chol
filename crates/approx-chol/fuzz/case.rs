use std::error::Error;

use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, UpperCsrRef,
};
use approx_chol::{factorize_with, Backend, Config, Factor, InnerParallelism};
use approx_chol::{Real, ScalingPolicy, SchurMode};
use serde::{Deserialize, Serialize};

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

/// `Config` in a `u8` per knob: `None` is an exact complement, under two copies sample once, a `u8` caps AC2.
#[derive(Debug, Serialize, Deserialize)]
pub struct Build {
    pub seed: u64,
    pub complement: Option<u8>,
    pub approximate: u8,
    pub dense_up_to: u8,
}

impl Build {
    pub fn config(&self) -> Config {
        let complement = (self.complement).map_or(SchurMode::Exact, |k| SchurMode::Approximate {
            split_merge: Some(k.into()),
        });
        Config {
            seed: self.seed,
            split_merge: Some(self.approximate.into()),
            backend: Backend::ExactBelow {
                max_dim: self.dense_up_to.into(),
            },
            complement,
        }
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
        Ok(FactorWidth1::new(
            self.n.into(),
            &self.rows,
            self.values.as_deref(),
        )?)
    }
}

impl<T: Real> Case<T> {
    pub fn factor(&self) -> Result<Factor<T>, Box<dyn Error>> {
        let policy = ScalingPolicy::new(self.tolerance, self.max_iterations.into())
            .ok_or("tolerance outside [0, 1)")?;
        let config = self.build.config();
        Ok(match &self.view {
            View::Csr(r) => {
                let csr = CsrRef::new(&r.row_ptrs, &r.columns, &r.values, r.n.into())?;
                factorize_with(csr, config, policy)?
            }
            View::UpperCsr(r) => {
                let csr = UpperCsrRef::new(&r.row_ptrs, &r.columns, &r.values, r.n.into())?;
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
        let solved = factor
            .solve(&b)
            .expect("a right-hand side of length n is accepted");
        let blocks = factor.into_blocks();
        // NaN scratch reaches the solution if any slot is read before written.
        for block in &blocks {
            let mut local = vec![T::one(); block.vertices().len()];
            let mut scratch = vec![T::nan(); block.scratch_len()];
            block
                .solve_local(&mut local, &mut scratch, InnerParallelism::Allow)
                .expect("buffers of the block's own lengths are accepted");
            let agrees = (block.vertices().zip(&local))
                .all(|(v, x)| solved[v].integer_decode() == x.integer_decode());
            assert!(agrees, "dirty scratch changed the solution");
        }
        Ok(blocks.len())
    }
}
