//! One component of `C` to a folded or signed SDDM part, eliminated side first; ported from within.

use super::cross::{Block, Cross};
use super::forest::Members;
use super::scaling::dominance_scaling;
use super::{Coordinates, Form, Gauge, NotScalable, Part, ScalingPolicy};

/// Evidence a dominance scaling exceeded tolerance or budget; the deficit was clamped.
pub(crate) struct Uncertified {
    pub(crate) iterations: usize,
    pub(crate) violation: f64,
}

/// What the input's shape already says about a component.
#[derive(Clone, Copy)]
pub(crate) enum Class {
    /// A 0/1 graph: balanced, dominant, zero surplus.
    Known,
    /// A Gram component: scaled, folded or covered, grounded until its columns say otherwise.
    General,
}

/// A component before conversion: its `C`, diagonal and global vertices, larger side first.
pub(crate) struct Oriented {
    pub(crate) cross: Cross,
    pub(crate) diagonal: Vec<f64>,
    pub(crate) globals: Vec<u32>,
    pub(crate) swapped: bool,
}

impl Oriented {
    /// `None` for a component whose diagonal is all zero: isolated zero rows solve to `0`.
    pub(crate) fn new(
        members: &Members,
        cross: &Cross,
        diagonal: &[f64],
        remap: &mut [u32],
        single: bool,
    ) -> Option<Self> {
        let n_left = cross.n_rows();
        let mut diagonal: Vec<f64> = (members.rows.iter().map(|&i| diagonal[i]))
            .chain(members.cols.iter().map(|&j| diagonal[n_left + j]))
            .collect();
        if diagonal.iter().all(|&d| d == 0.0) {
            return None;
        }
        let mut globals: Vec<u32> = (members.rows.iter().map(|&i| i as u32))
            .chain(members.cols.iter().map(|&j| (n_left + j) as u32))
            .collect();
        let mut cross = if single {
            cross.clone()
        } else {
            cross.extract(&members.rows, &members.cols, remap)
        };
        let swapped = cross.n_cols() > cross.n_rows();
        if swapped {
            diagonal.rotate_left(cross.n_rows());
            globals.rotate_left(cross.n_rows());
            cross = cross.swapped();
        }
        Some(Self {
            cross,
            diagonal,
            globals,
            swapped,
        })
    }

    pub(crate) fn convert(
        self,
        class: Class,
        policy: &ScalingPolicy,
    ) -> Result<(Part, Option<Uncertified>), NotScalable> {
        match class {
            Class::Known => self.known().map(|part| (part, None)),
            Class::General => self.general(policy),
        }
    }

    /// Skipping the signed machinery keeps a graph's part at two streaming passes.
    fn known(self) -> Result<Part, NotScalable> {
        let diagonal = adjacency_sums(&self.cross);
        if diagonal.iter().any(|d| !d.is_finite()) {
            return Err(NotScalable);
        }
        Ok(Part {
            surplus: vec![0.0; self.cross.n_local()],
            cross: self.cross,
            diagonal,
            globals: self.globals,
            gauge: Gauge::Floating,
            form: Form::Folded,
            coordinates: Coordinates::Canonical,
        })
    }

    /// Converts as grounded; the caller floats the part once its columns prove an alias.
    fn general(self, policy: &ScalingPolicy) -> Result<(Part, Option<Uncertified>), NotScalable> {
        let signs = folding_signs(&self.cross);
        let relaxation = dominance_scaling(&self.cross, &self.diagonal, policy)?;
        let n_rows = self.cross.n_rows();
        let (factors, form) = match signs {
            Some(signs) => (
                (signs.iter().zip(&relaxation.scales))
                    .map(|(&sign, &scale)| sign * scale)
                    .collect(),
                Form::Folded,
            ),
            None => {
                let mut factors = relaxation.scales;
                for factor in &mut factors[n_rows..] {
                    *factor = -*factor;
                }
                (factors, Form::Cover)
            }
        };
        let (part, clamped) = assemble(self, factors, form, policy)?;
        let violation = relaxation.violation.max(clamped);
        let uncertified = (violation > policy.tolerance).then_some(Uncertified {
            iterations: relaxation.iterations,
            violation,
        });
        Ok((part, uncertified))
    }
}

/// A folded part must be a Z-matrix, so a negative folded weight is an error.
fn assemble(
    oriented: Oriented,
    factors: Vec<f64>,
    form: Form,
    policy: &ScalingPolicy,
) -> Result<(Part, f64), NotScalable> {
    let Oriented {
        cross,
        diagonal,
        globals,
        ..
    } = oriented;
    let n_rows = cross.n_rows();
    let enforce_z = form == Form::Folded;
    let Cross { mut c, .. } = cross;
    for i in 0..n_rows {
        let span = c.span(i);
        for (&j, value) in c.indices[span.clone()].iter().zip(&mut c.data[span]) {
            let folded = -factors[i] * factors[n_rows + j as usize] * *value;
            if !folded.is_finite() || (enforce_z && folded < 0.0) {
                return Err(NotScalable);
            }
            *value = folded;
        }
    }
    let cross = Cross::new(c);
    let mut diagonal: Vec<f64> = (diagonal.iter().zip(&factors))
        .map(|(&entry, &factor)| entry * factor * factor)
        .collect();

    let canonical = (factors.iter().enumerate())
        .all(|(i, &factor)| factor == if i < n_rows { 1.0 } else { -1.0 });
    let coordinates = if canonical {
        Coordinates::Canonical
    } else {
        Coordinates::Scaled(factors)
    };

    // Clamp roundoff deficits and retain surplus as ground edges.
    let row_sums = magnitude_sums(&cross);
    let mut surplus = vec![0.0; cross.n_local()];
    let mut clamped = 0.0f64;
    for ((entry, &row_sum), slot) in diagonal.iter_mut().zip(&row_sums).zip(&mut surplus) {
        if !entry.is_finite() || *entry <= 0.0 {
            return Err(NotScalable);
        }
        let deficit = (row_sum - *entry) / *entry;
        if deficit > policy.tolerance {
            clamped = clamped.max(deficit);
        }
        *entry = entry.max(row_sum);
        *slot = (*entry - row_sum).max(0.0);
    }
    Ok((
        Part {
            cross,
            diagonal,
            surplus,
            globals,
            gauge: Gauge::Grounded,
            form,
            coordinates,
        },
        clamped,
    ))
}

impl Part {
    /// A floating part anchors nothing, so it drops its surplus.
    pub(crate) fn float(&mut self) {
        self.diagonal = magnitude_sums(&self.cross);
        self.surplus.fill(0.0);
        self.gauge = Gauge::Floating;
    }

    /// The factor a level's local slot carries.
    pub(crate) fn factor(&self, local: usize) -> f64 {
        let n_eliminated = self.cross.n_rows();
        match &self.coordinates {
            Coordinates::Canonical if local < n_eliminated => 1.0,
            Coordinates::Canonical => -1.0,
            Coordinates::Scaled(factors) => factors[local],
        }
    }
}

/// `None` when the component is frustrated: a negative cycle admits no signature.
fn folding_signs(cross: &Cross) -> Option<Vec<f64>> {
    let n = cross.n_local();
    let mut signs = vec![0.0; n];
    for root in 0..n {
        if signs[root] != 0.0 {
            continue;
        }
        signs[root] = 1.0;
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            for (j, value) in cross.neighbors(i) {
                let expected = -signs[i] * value.signum();
                if signs[j] == 0.0 {
                    signs[j] = expected;
                    stack.push(j);
                } else if signs[j] != expected {
                    return None;
                }
            }
        }
    }
    Some(signs)
}

fn adjacency_sums(cross: &Cross) -> Vec<f64> {
    (0..cross.n_local())
        .map(|i| cross.neighbors(i).map(|(_, v)| v).sum())
        .collect()
}

fn magnitude_sums(cross: &Cross) -> Vec<f64> {
    (0..cross.n_local())
        .map(|i| cross.neighbors(i).map(|(_, v)| v.abs()).sum())
        .collect()
}

/// Gremban cover of a signed part: SDDM, acting on the antisymmetric `[z, -z]` subspace as the part.
pub(crate) fn cover(part: &Part) -> (Cross, Vec<f64>, Vec<f64>) {
    let c = &part.cross.c;
    let (n_rows, n_cols) = (c.nrows, c.ncols);
    let mut indptr = Vec::with_capacity(2 * n_rows + 1);
    let mut indices = Vec::with_capacity(2 * c.nnz());
    let mut data = Vec::with_capacity(2 * c.nnz());
    indptr.push(0u32);
    for copy_shifted in [false, true] {
        for i in 0..n_rows {
            let span = c.span(i);
            // Same-sheet columns precede cross-sheet ones so each output row stays column-sorted.
            for column_shifted in [false, true] {
                let base = if column_shifted { n_cols as u32 } else { 0 };
                let select_negative = column_shifted != copy_shifted;
                for idx in span.clone() {
                    let value = c.data[idx];
                    if (value < 0.0) != select_negative {
                        continue;
                    }
                    indices.push(c.indices[idx] + base);
                    data.push(value.abs());
                }
            }
            indptr.push(indices.len() as u32);
        }
    }
    let cross = Cross::new(Block {
        indptr,
        indices,
        data,
        nrows: 2 * n_rows,
        ncols: 2 * n_cols,
    });
    let double = |values: &[f64]| {
        let (rows, cols) = values.split_at(n_rows);
        [rows, rows, cols, cols].concat()
    };
    (cross, double(&part.diagonal), double(&part.surplus))
}
