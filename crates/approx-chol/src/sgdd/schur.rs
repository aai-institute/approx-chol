//! Batch elimination of the eliminated side: its Schur complement on the kept side, ported from within.
//!
//! [`exact`] accumulates the true reduced rows; [`sampled`] samples each eliminated star's clique.

use super::cross::Cross;
use super::{compensated_sum, Gauge};
use crate::approx_chol::CliqueTreeSampler;
use crate::graph::CoreRows;
use rayon::prelude::*;

/// Multiply-adds below which splitting the kept rows costs more than it saves.
const PAR_ROW_SPLIT_THRESHOLD: usize = 100_000;

/// An SDDM in eliminated-major form: `[D_e, −C; −Cᵀ, D_k]`, surplus inside the diagonal.
pub(crate) struct Eliminated<'a> {
    pub(crate) cross: &'a Cross,
    pub(crate) diagonal: &'a [f64],
    pub(crate) surplus: &'a [f64],
    pub(crate) gauge: Gauge,
    pub(crate) inv_diagonal: Vec<f64>,
}

impl Eliminated<'_> {
    fn n_eliminated(&self) -> usize {
        self.cross.n_rows()
    }

    fn n_kept(&self) -> usize {
        self.cross.n_cols()
    }

    /// The reduced system's vertex count: the kept side plus a ground when grounded.
    pub(crate) fn n_reduced(&self) -> usize {
        self.n_kept() + usize::from(self.gauge == Gauge::Grounded)
    }
}

struct UpperRow {
    columns: Vec<u32>,
    values: Vec<f64>,
}

/// Exact Schur complement with an explicit ground row when grounded.
pub(crate) fn exact(system: &Eliminated<'_>) -> CoreRows<f64> {
    let n_keep = system.n_kept();
    let row = |i: usize, work: &mut Vec<f64>, touched: &mut Vec<usize>| {
        schur_row(system, i, work, touched);
        let result = extract_row(i, work, touched);
        touched.clear();
        result
    };
    let flops: usize = (system.cross.c.indptr.windows(2))
        .map(|bounds| {
            let width = (bounds[1] - bounds[0]) as usize;
            width * (width + 1) / 2
        })
        .sum();
    let rows: Vec<UpperRow> = if flops < PAR_ROW_SPLIT_THRESHOLD {
        let mut work = vec![0.0f64; n_keep];
        let mut touched = Vec::new();
        (0..n_keep)
            .map(|i| row(i, &mut work, &mut touched))
            .collect()
    } else {
        (0..n_keep)
            .into_par_iter()
            .map_init(
                || (vec![0.0f64; n_keep], Vec::new()),
                |(work, touched), i| row(i, work, touched),
            )
            .collect()
    };
    let (indptr, indices, data) = full_rows(rows, n_keep);
    let surplus = reduced_surplus(system);
    explicit_laplacian(&indptr, &indices, &data, &surplus, system.gauge)
}

fn schur_row(system: &Eliminated<'_>, i: usize, work: &mut [f64], touched: &mut Vec<usize>) {
    let n_e = system.n_eliminated();
    work[i] = system.diagonal[n_e + i];
    touched.push(i);
    let c = &system.cross.c;
    for (k, w) in system.cross.ct.row(i) {
        let inv = system.inv_diagonal[k];
        let span = c.span(k);
        let columns = &c.indices[span.clone()];
        let upper_start = columns.partition_point(|&j| (j as usize) < i);
        for position in span.start + upper_start..span.end {
            let j = c.indices[position] as usize;
            let v = c.data[position];
            if work[j] == 0.0 && j != i {
                touched.push(j);
            }
            work[j] -= w * v * inv;
        }
    }
}

/// Keeps a numerically zero diagonal so the row stays present.
fn extract_row(i: usize, work: &mut [f64], touched: &mut [usize]) -> UpperRow {
    touched.sort_unstable();
    let mut columns = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for &j in touched.iter() {
        let v = work[j];
        if v != 0.0 || j == i {
            columns.push(j as u32);
            values.push(v);
        }
        work[j] = 0.0;
    }
    UpperRow { columns, values }
}

/// Full rows from diagonal-and-upper parts; row layout `[mirrored lower | diagonal and upper]`.
fn full_rows(rows: Vec<UpperRow>, n: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut indptr = vec![0u32; n + 1];
    for (i, row) in rows.iter().enumerate() {
        indptr[i + 1] += row.columns.len() as u32;
        for &j in &row.columns {
            if j as usize != i {
                indptr[j as usize + 1] += 1;
            }
        }
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let nnz = indptr[n] as usize;
    let mut indices = vec![0u32; nnz];
    let mut data = vec![0.0f64; nnz];
    let mut lower = indptr[..n].to_vec();
    for (i, row) in rows.into_iter().enumerate() {
        let upper_start = indptr[i + 1] as usize - row.columns.len();
        indices[upper_start..upper_start + row.columns.len()].copy_from_slice(&row.columns);
        data[upper_start..upper_start + row.values.len()].copy_from_slice(&row.values);
        for (&j, &v) in row.columns.iter().zip(&row.values) {
            if j as usize != i {
                let position = lower[j as usize] as usize;
                indices[position] = i as u32;
                data[position] = v;
                lower[j as usize] += 1;
            }
        }
    }
    (indptr, indices, data)
}

fn reduced_surplus(system: &Eliminated<'_>) -> Vec<f64> {
    let n_e = system.n_eliminated();
    if system.gauge == Gauge::Floating {
        return vec![0.0; system.n_kept()];
    }
    let scaled: Vec<f64> = (system.inv_diagonal.iter())
        .zip(&system.surplus[..n_e])
        .map(|(&inv, &surplus)| inv * surplus)
        .collect();
    let mut surplus = vec![0.0; system.n_kept()];
    system
        .cross
        .ct
        .multiply_add(&scaled, &system.surplus[n_e..], &mut surplus);
    surplus
}

/// The principal rows' diagonal becomes their adjacency plus surplus, which joins an explicit ground.
fn explicit_laplacian(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    surplus: &[f64],
    gauge: Gauge,
) -> CoreRows<f64> {
    let n_keep = indptr.len() - 1;
    let ground = n_keep as u32;
    let grounded = gauge == Gauge::Grounded;
    let mut out_ptr = vec![0u32];
    let mut out_idx = Vec::with_capacity(indices.len() + 2 * n_keep);
    let mut out_val = Vec::with_capacity(indices.len() + 2 * n_keep);
    for (i, &row_surplus) in surplus.iter().enumerate() {
        let span = indptr[i] as usize..indptr[i + 1] as usize;
        let mut adjacency = 0.0;
        let mut diagonal_position = None;
        for (&j, &value) in indices[span.clone()].iter().zip(&data[span]) {
            if j as usize == i {
                diagonal_position = Some(out_idx.len());
                out_idx.push(j);
                out_val.push(0.0);
            } else {
                adjacency -= value;
                out_idx.push(j);
                out_val.push(value);
            }
        }
        let diagonal_position = diagonal_position.expect("an exact Schur row keeps its diagonal");
        out_val[diagonal_position] = adjacency + row_surplus;
        if grounded && row_surplus > 0.0 {
            out_idx.push(ground);
            out_val.push(-row_surplus);
        }
        out_ptr.push(out_idx.len() as u32);
    }
    if grounded {
        for (i, &value) in surplus.iter().enumerate() {
            if value > 0.0 {
                out_idx.push(i as u32);
                out_val.push(-value);
            }
        }
        out_idx.push(ground);
        out_val.push(compensated_sum(surplus));
        out_ptr.push(out_idx.len() as u32);
    }
    CoreRows::from_full_rows(&out_ptr, &out_idx, &out_val)
}

/// Undirected fill edge `(lo, hi, weight)` with `lo < hi`.
type Edge = (u32, u32, f64);

/// One rayon task's sampling state; the sampler carries scratch of its own.
struct StarWorkspace {
    edges: Vec<Edge>,
    entries: Vec<(u32, f64)>,
    sampler: CliqueTreeSampler,
}

/// Sampled Schur complement as a Laplacian; a ground stays an ordinary final vertex.
pub(crate) fn sampled(system: &Eliminated<'_>, seed: u64, split: u32) -> CoreRows<f64> {
    let n_kept = system.n_kept();
    let n_e = system.n_eliminated();
    let c = &system.cross.c;
    let ground = (system.gauge == Gauge::Grounded).then_some(n_kept as u32);
    // The total order fixes per-`(lo, hi)` summation, so the result is scheduling-independent.
    let mut chunks = (0..n_e)
        .into_par_iter()
        .fold(
            || StarWorkspace {
                edges: Vec::new(),
                entries: Vec::new(),
                sampler: CliqueTreeSampler::new(seed, Some(split)),
            },
            |mut work, k| {
                let span = c.span(k);
                if !span.is_empty() {
                    work.entries.clear();
                    for (&col, &w) in c.indices[span.clone()].iter().zip(&c.data[span]) {
                        work.entries.push((col, w));
                    }
                    if let Some(ground) = ground {
                        if system.surplus[k] > 0.0 {
                            work.entries.push((ground, system.surplus[k]));
                        }
                    }
                    if work.entries.len() > 1 {
                        work.sampler
                            .sample(k as u64, &work.entries, &mut work.edges);
                    }
                }
                work
            },
        )
        .map(|work| work.edges)
        .collect::<Vec<_>>();
    if let Some(ground) = ground {
        chunks.push(
            (system.surplus[n_e..].iter().enumerate())
                .filter(|(_, surplus)| **surplus > 0.0)
                .map(|(i, &surplus)| (i as u32, ground, surplus))
                .collect(),
        );
    }
    let edges = sort_and_merge(chunks, n_kept);
    CoreRows::from_edges(system.n_reduced(), &edges)
}

/// Sort into total `(lo, hi, weight)` order; the weight tiebreak makes the Schur reproducible.
fn sort_and_merge(chunks: Vec<Vec<Edge>>, n_kept: usize) -> Vec<Edge> {
    let total = chunks.iter().map(Vec::len).sum::<usize>();
    if total <= 1 {
        return chunks.concat();
    }
    // Same order either way; under two edges per kept vertex the parallel comparison sort wins.
    let dense = total >= 2 * n_kept;
    let mut edges = if dense {
        counting_sort_by_lo(&chunks, n_kept)
    } else {
        chunks.concat()
    };
    drop(chunks);
    if dense {
        let by_hi_weight = |a: &Edge, b: &Edge| a.1.cmp(&b.1).then_with(|| a.2.total_cmp(&b.2));
        edges
            .par_chunk_by_mut(|a, b| a.0 == b.0)
            .for_each(|run| run.sort_unstable_by(by_hi_weight));
    } else {
        edges.par_sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then_with(|| a.2.total_cmp(&b.2))
        });
    }
    let mut write = 0;
    for read in 1..edges.len() {
        if edges[write].0 == edges[read].0 && edges[write].1 == edges[read].1 {
            edges[write].2 += edges[read].2;
        } else {
            write += 1;
            edges[write] = edges[read];
        }
    }
    edges.truncate(write + 1);
    edges
}

/// Stable counting sort by `lo`; the ground has the maximum id, so it only lands on `hi`.
fn counting_sort_by_lo(chunks: &[Vec<Edge>], n_kept: usize) -> Vec<Edge> {
    let mut cursors = vec![0usize; n_kept + 1];
    for chunk in chunks {
        for e in chunk {
            cursors[e.0 as usize + 1] += 1;
        }
    }
    for i in 1..cursors.len() {
        cursors[i] += cursors[i - 1];
    }
    let mut out = vec![(0u32, 0u32, 0.0f64); cursors[n_kept]];
    for chunk in chunks {
        for &e in chunk {
            let cursor = &mut cursors[e.0 as usize];
            out[*cursor] = e;
            *cursor += 1;
        }
    }
    out
}
