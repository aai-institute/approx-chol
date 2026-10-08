//! Per part: the eliminated side's diagonal, the kernel's factor of the rest, and the lift back to `A`.

use super::cross::Cross;
use super::schur::{self, Eliminated};
use super::{compensated_sum, convert, Coordinates, Form, Gauge, Part, SchurMode, Sgdd};
use crate::approx_chol::{factor_core, Block};
use crate::{Backend, Config, Error, ExactFailure, Fallback};
use rayon::prelude::*;

/// Applies `Ã⁺` per part in `A`'s coordinates; vertices in no part solve to `0`.
#[derive(Clone, Debug)]
pub struct SgddFactor {
    n: usize,
    parts: Vec<FactorPart>,
    fallbacks: Vec<Fallback>,
}

#[derive(Clone, Debug)]
struct FactorPart {
    globals: Vec<u32>,
    /// The single (signed, for a cover) `C`, eliminated by kept.
    cross: Cross,
    inv_diagonal: Vec<f64>,
    reduced: Block<f64>,
    gauge: Gauge,
    form: Form,
    coordinates: Coordinates,
    /// Max-normalized `R⁻¹·1` of a scaled floating part, so the solve returns `A⁺b`.
    null: Option<Vec<f64>>,
}

fn subtract_mean(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mean = compensated_sum(values) / values.len() as f64;
    for value in values.iter_mut() {
        *value -= mean;
    }
}

fn project_out(values: &mut [f64], null: &[f64]) {
    let zz: f64 = null.iter().map(|z| z * z).sum();
    let zv: f64 = null.iter().zip(values.iter()).map(|(z, v)| z * v).sum();
    for (value, z) in values.iter_mut().zip(null) {
        *value -= z * (zv / zz);
    }
}

fn inverse_diagonal(diagonal: &[f64]) -> Vec<f64> {
    diagonal.iter().map(|&d| 1.0 / d).collect()
}

/// Exact Schur and a dense factor when the kept side is small, as within routes.
fn factor_reduced(
    system: &Eliminated<'_>,
    config: &Config,
    mode: SchurMode,
) -> Result<(Block<f64>, Option<Fallback>), Error> {
    let threshold = match config.backend {
        Backend::ExactBelow { max_dim, .. } => max_dim,
        Backend::Approximate => 0,
    };
    let with = |on_failure| Config {
        backend: Backend::ExactBelow {
            max_dim: threshold,
            on_failure,
        },
        ..*config
    };
    let n_kept = system.cross.n_cols();
    let exact = (threshold > 0 && n_kept <= threshold).then(|| schur::exact(system));
    let wrap = |cholesky| match system.gauge {
        Gauge::Grounded => Block::Grounded(cholesky),
        Gauge::Floating => Block::Floating(cholesky),
    };
    if let Some(exact) = &exact {
        match factor_core(with(ExactFailure::Error), exact) {
            Err(Error::DenseFactorizationFailed(_)) => {}
            result => return result.map(|(cholesky, fallback)| (wrap(cholesky), fallback)),
        }
    }
    let complement = match mode {
        SchurMode::Sampled { seed, split } => schur::sampled(system, seed, split),
        SchurMode::Exact => exact.unwrap_or_else(|| schur::exact(system)),
    };
    factor_core(with(ExactFailure::FallBackToApproximate), &complement)
        .map(|(cholesky, fallback)| (wrap(cholesky), fallback))
}

impl FactorPart {
    fn build(
        part: Part,
        config: &Config,
        mode: SchurMode,
    ) -> Result<(Self, Option<Fallback>), Error> {
        let n_e = part.cross.n_rows();
        let inv_diagonal = inverse_diagonal(&part.diagonal[..n_e]);
        let (reduced, fallback) = match part.form {
            Form::Folded => factor_reduced(
                &Eliminated {
                    cross: &part.cross,
                    diagonal: &part.diagonal,
                    surplus: &part.surplus,
                    gauge: part.gauge,
                    inv_diagonal: inv_diagonal.clone(),
                },
                config,
                mode,
            )?,
            Form::Cover => {
                let (cross, diagonal, surplus) = convert::cover(&part);
                let system = Eliminated {
                    cross: &cross,
                    inv_diagonal: inverse_diagonal(&diagonal[..cross.n_rows()]),
                    diagonal: &diagonal,
                    surplus: &surplus,
                    gauge: part.gauge,
                };
                let (block, fallback) = factor_reduced(&system, config, mode)?;
                // The antisymmetric embed is consistent for either gauge, so no ground is pinned.
                let cholesky = match block {
                    Block::Grounded(cholesky) | Block::Floating(cholesky) => cholesky,
                };
                (Block::Floating(cholesky), fallback)
            }
        };
        let null = match (&part.coordinates, part.gauge, part.form) {
            (Coordinates::Scaled(factors), Gauge::Floating, Form::Folded) => {
                let peak = factors.iter().fold(0.0f64, |peak, f| peak.max(f.abs()));
                Some(factors.iter().map(|f| f / peak).collect())
            }
            _ => None,
        };
        Ok((
            Self {
                globals: part.globals,
                cross: part.cross,
                inv_diagonal,
                reduced,
                gauge: part.gauge,
                form: part.form,
                coordinates: part.coordinates,
                null,
            },
            fallback,
        ))
    }

    fn fold(&self, values: &mut [f64]) {
        let n_e = self.cross.n_rows();
        match &self.coordinates {
            Coordinates::Canonical => values[n_e..].iter_mut().for_each(|v| *v = -*v),
            Coordinates::Scaled(factors) => {
                values.iter_mut().zip(factors).for_each(|(v, f)| *v *= f);
            }
        }
    }

    /// `x` arrives as the part's right-hand side and leaves as its solution, both in `A`'s coordinates.
    fn solve(&self, x: &mut [f64]) {
        let n_e = self.cross.n_rows();
        let n_k = self.cross.n_cols();
        let floating = self.gauge == Gauge::Floating && self.form == Form::Folded;
        if let Some(null) = &self.null {
            project_out(x, null);
        }
        self.fold(x);
        if floating {
            subtract_mean(x);
        }

        let (rhs_e, rhs_k) = x.split_at_mut(n_e);
        for (value, inv) in rhs_e.iter_mut().zip(&self.inv_diagonal) {
            *value *= inv;
        }
        let mut slots = vec![0.0; self.reduced.slots()];
        self.cross.ct.multiply_add(rhs_e, rhs_k, &mut slots[..n_k]);
        match self.form {
            Form::Folded => {
                if floating {
                    subtract_mean(&mut slots[..n_k]);
                }
                self.reduced.solve(&mut slots);
            }
            Form::Cover => {
                let (positive, negative) = slots.split_at_mut(n_k);
                for (minus, &plus) in negative[..n_k].iter_mut().zip(positive.iter()) {
                    *minus = -plus;
                }
                self.reduced.solve(&mut slots);
                for i in 0..n_k {
                    slots[i] = 0.5 * (slots[i] - slots[n_k + i]);
                }
            }
        }
        rhs_k.copy_from_slice(&slots[..n_k]);

        // Back-substitution reads the gauged kept block.
        for (i, out) in rhs_e.iter_mut().enumerate() {
            let sum: f64 = self.cross.c.row(i).map(|(j, v)| v * rhs_k[j]).sum();
            *out += self.inv_diagonal[i] * sum;
        }
        if floating {
            subtract_mean(x);
        }
        self.fold(x);
        if let Some(null) = &self.null {
            project_out(x, null);
        }
    }
}

impl Sgdd {
    /// Factors every part, in parallel; `config` routes each reduced system as [`Builder`](crate::Builder) routes a block.
    pub fn factorize(self, config: &Config, mode: SchurMode) -> Result<SgddFactor, Error> {
        let built: Vec<(FactorPart, Option<Fallback>)> = (self.parts.into_par_iter())
            .map(|part| FactorPart::build(part, config, mode))
            .collect::<Result<_, _>>()?;
        let (parts, fallbacks): (Vec<_>, Vec<_>) = built.into_iter().unzip();
        Ok(SgddFactor {
            n: self.n,
            parts,
            fallbacks: fallbacks.into_iter().flatten().collect(),
        })
    }
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

    /// `x = Ã⁺ b`; panics unless `b` has length [`n`](Self::n).
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        assert_eq!(b.len(), self.n, "rhs length must equal n");
        let mut x = vec![0.0; self.n];
        let mut local = Vec::new();
        for part in &self.parts {
            local.clear();
            local.extend(part.globals.iter().map(|&g| b[g as usize]));
            part.solve(&mut local);
            for (&g, &value) in part.globals.iter().zip(&local) {
                x[g as usize] = value;
            }
        }
        x
    }
}
