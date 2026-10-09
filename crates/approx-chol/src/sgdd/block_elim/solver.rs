use std::borrow::Borrow;
use std::sync::Arc;

use crate::{Error, ExactFailure, Factor, SolveError};
use rayon::prelude::*;

use super::{ApproxSchurConfig, LocalSolverConfig};
use crate::sgdd::cross_tab::CrossTab;
use crate::sgdd::csr_block::{CsrBlock, PAR_SPMV_THRESHOLD};
use crate::sgdd::factor_pairs::{CoordinateMap, Grounding, LocalComponent, MatrixForm, SddmMatrix};
use crate::sgdd::SchurMode;

use super::compensated_sum;
use super::factor::{ReducedFactor, ReducedSystem};
use super::schur;

/// Minimum number of rows to trigger parallel back-substitution.
const PAR_BACKSUB_THRESHOLD: usize = 10_000;
const PAR_BACKSUB_CHUNK: usize = 4096;

/// Subtract the mean of `slice[..n]` from those `n` elements.
#[inline]
fn subtract_mean(slice: &mut [f64], n: usize) {
    if n == 0 {
        return;
    }
    let mean = compensated_sum(&slice[..n]) / n as f64;
    for val in slice[..n].iter_mut() {
        *val -= mean;
    }
}

#[inline]
fn scale_by_diag_in_place(slice: &mut [f64], diagonal: &[f64]) {
    debug_assert!(diagonal.len() >= slice.len());
    for (value, &scale) in slice.iter_mut().zip(diagonal.iter()) {
        *value *= scale;
    }
}

/// Back-substitute for the eliminated block from a pre-scaled RHS.
fn backsub_block_from_scaled_rhs(
    sol_output: &mut [f64],
    scaled_rhs: &[f64],
    cross_matrix: &CsrBlock,
    inv_diag: &[f64],
    sol_source: &[f64],
    allow_inner_parallelism: bool,
) {
    let n = sol_output.len();
    debug_assert!(scaled_rhs.len() >= n);
    let recover = |i: usize| -> f64 {
        let start = cross_matrix.indptr[i] as usize;
        let end = cross_matrix.indptr[i + 1] as usize;
        let mut sum = 0.0;
        for idx in start..end {
            let j = cross_matrix.indices[idx] as usize;
            sum += cross_matrix.data[idx] * sol_source[j];
        }
        scaled_rhs[i] + (inv_diag[i] * sum)
    };
    if n > PAR_BACKSUB_THRESHOLD && allow_inner_parallelism {
        sol_output
            .par_chunks_mut(PAR_BACKSUB_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * PAR_BACKSUB_CHUNK;
                for (local_i, si) in chunk.iter_mut().enumerate() {
                    *si = recover(row_start + local_i);
                }
            });
    } else {
        for (i, out) in sol_output.iter_mut().enumerate() {
            *out = recover(i);
        }
    }
}

/// Local subdomain solver using block elimination on the bipartite SDDM.
#[derive(Clone, Debug)]
pub(crate) struct BlockElimSolver {
    /// Bipartite Gramian `C` and `Cᵀ`; diagonals are folded in at build time and not retained.
    cross_tab: Arc<CrossTab>,
    /// `1 / D_elim[k]` for the eliminated (larger) diagonal block.
    inv_diag_elim: Vec<f64>,
    /// Reduced-system factor backend.
    reduced_factor: ReducedFactor,
    /// Internal DOF count; the matrix is single-sized, a cover living in `reduced_factor`.
    n_internal: usize,
    /// Original-to-SDDM coordinate map.
    coordinates: CoordinateMap,
}

/// An eliminated-major SDDM plus its eliminated-diagonal reciprocals, which it does not carry.
struct Eliminated {
    matrix: SddmMatrix,
    inv_diagonal: Vec<f64>,
}

impl Eliminated {
    fn new(matrix: SddmMatrix) -> Result<Self, Error> {
        debug_assert!(
            matrix.n_eliminated() >= matrix.n_kept(),
            "component is not eliminated-major"
        );
        let inv_diagonal = matrix.diagonal[..matrix.n_eliminated()]
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d > 0.0 {
                    Ok(1.0 / d)
                } else {
                    Err(Error::NotDiagonallyDominant { row: i })
                }
            })
            .collect::<Result<Vec<f64>, Error>>()?;
        Ok(Self {
            matrix,
            inv_diagonal,
        })
    }

    /// Fold of the signed matrix's Gremban cover; transient, dropped once its complement is built.
    fn cover(&self) -> Result<Self, Error> {
        Self::new(assemble_bipartite_cover(&self.matrix))
    }

    /// A dense Cholesky spends nothing on sparsity, so it entails the exact complement.
    fn factor_reduced(
        fold: impl Borrow<Self>,
        config: &LocalSolverConfig,
    ) -> Result<Factor, Error> {
        let this = fold.borrow();
        let exact_below = config.dense_threshold;
        let exact = (exact_below > 0 && this.matrix.n_kept() <= exact_below)
            .then(|| schur::exact_for_factor(&this.matrix, &this.inv_diagonal));
        if let Some(exact) = &exact {
            match factor_complement(exact, config, ExactFailure::Error) {
                Err(Error::DenseFactorizationFailed { .. }) => {}
                result => return result,
            }
        }
        let complement = match config.schur {
            SchurMode::Sampled { seed, split } => {
                schur::sampled(&this.matrix, &ApproxSchurConfig { seed, split })
            }
            SchurMode::Exact => {
                exact.unwrap_or_else(|| schur::exact_for_factor(&this.matrix, &this.inv_diagonal))
            }
        };
        // An owned fold is the transient cover; it is freed before the factor's fill is allocated.
        drop(fold);
        factor_complement(&complement, config, ExactFailure::FallBackToApproximate)
    }
}

fn factor_complement(
    complement: &ReducedSystem,
    config: &LocalSolverConfig,
    on_failure: ExactFailure,
) -> Result<Factor, Error> {
    complement.factor(config.to_approx_chol(on_failure))
}

/// Gremban cover: SDDM, and acts on the antisymmetric `[z, -z]` subspace as the original.
fn assemble_bipartite_cover(matrix: &SddmMatrix) -> SddmMatrix {
    let c = &matrix.cross_tab.c;
    let n_rows = c.nrows;
    let n_cols = c.ncols;
    let n_cols_u32 = u32::try_from(n_cols).expect("cover columns exceed u32::MAX");
    // Both doubled sizes must fit u32: the cross-sheet shift emits columns up to `2*n_cols - 1`.
    u32::try_from(2 * n_rows).expect("doubled cover rows exceed u32::MAX");
    u32::try_from(2 * n_cols).expect("doubled cover columns exceed u32::MAX");

    let mut indptr = Vec::with_capacity(2 * n_rows + 1);
    let mut indices = Vec::with_capacity(2 * c.nnz());
    let mut data = Vec::with_capacity(2 * c.nnz());
    indptr.push(0u32);
    for copy_shifted in [false, true] {
        for i in 0..n_rows {
            let start = c.indptr[i] as usize;
            let end = c.indptr[i + 1] as usize;
            // Same-sheet columns precede cross-sheet ones so each output row stays column-sorted.
            for column_shifted in [false, true] {
                let column_base = if column_shifted { n_cols_u32 } else { 0 };
                let select_negative = column_shifted != copy_shifted;
                for idx in start..end {
                    let value = c.data[idx];
                    if (value < 0.0) != select_negative {
                        continue;
                    }
                    indices.push(c.indices[idx] + column_base);
                    data.push(value.abs());
                }
            }
            indptr.push(u32::try_from(indices.len()).expect("cover nonzeros exceed u32::MAX"));
        }
    }
    let cover_c = CsrBlock {
        indptr,
        indices,
        data,
        nrows: 2 * n_rows,
        ncols: 2 * n_cols,
    };
    SddmMatrix {
        cross_tab: CrossTab::new(cover_c),
        diagonal: double_for_cover(&matrix.diagonal, n_rows),
        ground_edges: double_for_cover(&matrix.ground_edges, n_rows),
        grounding: matrix.grounding,
    }
}

/// Each Gremban sheet copies every vertex, so a per-vertex array doubles within each side.
fn double_for_cover(values: &[f64], n_rows: usize) -> Vec<f64> {
    let (rows, cols) = values.split_at(n_rows);
    [rows, rows, cols, cols].concat()
}

impl BlockElimSolver {
    /// Size of the block the elimination removes, the leading one.
    fn n_eliminated(&self) -> usize {
        self.cross_tab.n_rows()
    }

    /// Size of the block the reduced factor solves.
    fn n_kept(&self) -> usize {
        self.cross_tab.n_cols()
    }

    pub(crate) fn new(
        cross_tab: impl Into<Arc<CrossTab>>,
        inv_diag_elim: Vec<f64>,
        reduced_factor: ReducedFactor,
        coordinates: CoordinateMap,
    ) -> Self {
        let cross_tab = cross_tab.into();
        // Every solve reads the transpose, so it is built here rather than inside the first.
        cross_tab.ct();
        debug_assert_eq!(reduced_factor.dimension(), cross_tab.n_cols());
        let n_internal = cross_tab.n_local();
        Self {
            cross_tab,
            inv_diag_elim,
            reduced_factor,
            n_internal,
            coordinates,
        }
    }

    /// A signed component keeps only the cover's factor, keeping the matrix single-sized (#91).
    pub(crate) fn build(
        component: LocalComponent,
        config: &LocalSolverConfig,
    ) -> Result<Self, Error> {
        let LocalComponent {
            matrix,
            form,
            coordinates,
        } = component;
        let eliminated = Eliminated::new(matrix)?;

        let factor = match form {
            MatrixForm::Laplacian => ReducedFactor::Direct {
                factor: Eliminated::factor_reduced(&eliminated, config)?,
                grounding: eliminated.matrix.grounding,
            },
            // Surplus survives the cover, so it grounds as the signed matrix did.
            MatrixForm::SignedPendingCover => ReducedFactor::Cover {
                inner: Eliminated::factor_reduced(eliminated.cover()?, config)?,
                m: eliminated.matrix.n_kept(),
            },
        };

        Ok(BlockElimSolver::new(
            eliminated.matrix.cross_tab,
            eliminated.inv_diagonal,
            factor,
            coordinates,
        ))
    }

    /// Components arrive oriented, so the row block is the eliminated side.
    fn eliminate_and_recover(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        let n = self.n_internal;
        let (n_elim, n_keep) = (self.n_eliminated(), self.n_kept());

        scale_by_diag_in_place(&mut rhs[..n_elim], &self.inv_diag_elim);

        {
            let (main, scratch) = rhs.split_at_mut(n);
            self.cross_tab.ct().spmv_assign_add(
                &main[..n_elim],
                &main[n_elim..],
                &mut scratch[..n_keep],
                allow_inner_parallelism,
            );
        }
        // A grounded factor and a cover each fix their own gauge; only a floating one needs this.
        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(&mut rhs[n..], n_keep);
        }

        // The backend's ground slot past the kept block is written before it is read.
        let slots = self.reduced_factor.slots();
        let reduced = n_elim..n_elim + slots;
        debug_assert!(
            reduced.end <= sol.len() && n + slots <= rhs.len(),
            "reduced solve and its RHS copy must fit sol and rhs",
        );
        sol[reduced.clone()].copy_from_slice(&rhs[n..n + slots]);
        let embed = &mut rhs[n + slots..];
        self.reduced_factor
            .solve_in_place(&mut sol[reduced], embed)?;

        let (sol_output, sol_source) = sol.split_at_mut(n_elim);
        let sol_source = &sol_source[..n_keep];
        backsub_block_from_scaled_rhs(
            sol_output,
            &rhs[..n_elim],
            &self.cross_tab.c,
            &self.inv_diag_elim,
            sol_source,
            allow_inner_parallelism,
        );
        Ok(())
    }
}

impl BlockElimSolver {
    pub(crate) fn fallbacks(&self) -> &[crate::Fallback] {
        self.reduced_factor.fallbacks()
    }

    pub(crate) fn n_local(&self) -> usize {
        self.n_internal
    }

    pub(crate) fn scratch_size(&self) -> usize {
        self.n_internal + self.reduced_factor.slots() + self.reduced_factor.scratch_len()
    }

    pub(crate) fn inner_parallelism_work_estimate(&self) -> usize {
        let max_rows = self.cross_tab.n_rows().max(self.cross_tab.n_cols());
        if max_rows <= PAR_BACKSUB_THRESHOLD.max(PAR_SPMV_THRESHOLD) {
            return 0;
        }

        let cross_nnz = self.cross_tab.c.nnz();
        (2 * cross_nnz) + self.n_internal
    }

    pub(crate) fn solve_local(
        &self,
        rhs: &mut [f64],
        sol: &mut [f64],
        allow_inner_parallelism: bool,
    ) -> Result<(), SolveError> {
        let n = self.n_internal;
        let n_eliminated = self.n_eliminated();

        self.coordinates.fold(&mut rhs[..n], n_eliminated);
        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(rhs, n);
        }

        self.eliminate_and_recover(rhs, sol, allow_inner_parallelism)?;

        if self.reduced_factor.grounding() == Some(Grounding::Floating) {
            subtract_mean(sol, n);
        }
        self.coordinates.unfold(&mut sol[..n], n_eliminated);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, Config};

    /// SCI's scaling of a component whose vertex 3 couples through a cell cancelled to 1.7e-16.
    fn scaled_near_inverse_epsilon() -> LocalComponent {
        let c = CsrBlock {
            indptr: vec![0, 1, 3],
            indices: vec![0, 0, 1],
            data: vec![
                0.03390196555552335,
                0.03390196555552337,
                1.6147876412209096e-33,
            ],
            nrows: 2,
            ncols: 2,
        };
        LocalComponent {
            matrix: SddmMatrix {
                cross_tab: CrossTab::eager(c),
                diagonal: vec![
                    0.03390196555552335,
                    0.03390196555552338,
                    1.0,
                    2.7733391199176196e-32,
                ],
                ground_edges: vec![
                    0.0,
                    1.3877787807814457e-17,
                    0.9321960688889532,
                    2.611860355795529e-32,
                ],
                grounding: Grounding::Grounded,
            },
            form: MatrixForm::Laplacian,
            coordinates: CoordinateMap::Scaled(
                vec![
                    0.05822539442161242,
                    -0.05822539442161245,
                    1.0,
                    -1.6653345369377348e-16,
                ]
                .into_boxed_slice(),
            ),
        }
    }

    #[test]
    fn a_grounded_solve_survives_a_scale_range_near_inverse_epsilon() {
        // The unscaled block: diag(10, 10, 1, 1) with the three cells it couples.
        let apply = |x: &[f64]| {
            let (c00, c10, c11) = (
                -0.5822539442161242,
                0.5822539442161243,
                -1.6653345369377348e-16,
            );
            [
                10.0 * x[0] + c00 * x[2],
                10.0 * x[1] + c10 * x[2] + c11 * x[3],
                x[2] + c00 * x[0] + c10 * x[1],
                x[3] + c11 * x[1],
            ]
        };
        let b = [0.3, -0.2, 0.1, 0.4];
        for (schur, max_dim) in [
            (SchurMode::Exact, usize::MAX),
            (SchurMode::Sampled { seed: 0, split: 1 }, 0),
        ] {
            let approx_chol = Config {
                backend: Backend::ExactBelow {
                    max_dim,
                    on_failure: ExactFailure::FallBackToApproximate,
                },
                ..Config::default()
            };
            let config = LocalSolverConfig {
                approx_chol,
                schur,
                dense_threshold: max_dim,
            };
            let solver = BlockElimSolver::build(scaled_near_inverse_epsilon(), &config).unwrap();
            let mut rhs = b.to_vec();
            rhs.resize(solver.scratch_size(), 0.0);
            let mut sol = vec![0.0; solver.scratch_size()];
            solver.solve_local(&mut rhs, &mut sol, false).unwrap();
            let residual = apply(&sol[..4])
                .iter()
                .zip(&b)
                .map(|(ax, b)| (ax - b).abs())
                .fold(0.0, f64::max);
            assert!(residual < 1e-12, "{schur:?}: residual {residual:e}");
        }
    }
}
