//! Proptest strategies and reference helpers for Laplacian/SDDM CSR input.
//! Each suite that includes this file uses a subset.
#![allow(dead_code)]

use proptest::prelude::*;
use std::collections::VecDeque;

pub type LaplacianCsr = (Vec<u32>, Vec<u32>, Vec<f64>, u32);

pub fn build_laplacian_csr(n: usize, edge_weights: &[u8]) -> LaplacianCsr {
    let mut dense = vec![0.0_f64; n * n];
    let mut edge_pos = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let w = edge_weights[edge_pos] as f64;
            edge_pos += 1;
            if w <= 0.0 {
                continue;
            }
            dense[i * n + j] -= w;
            dense[j * n + i] -= w;
            dense[i * n + i] += w;
            dense[j * n + j] += w;
        }
    }

    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0u32);
    for i in 0..n {
        for j in 0..n {
            let value = dense[i * n + j];
            if i == j || value != 0.0 {
                col_indices.push(j as u32);
                values.push(value);
            }
        }
        row_ptrs.push(col_indices.len() as u32);
    }

    (row_ptrs, col_indices, values, n as u32)
}

/// `A'[p[i]][p[j]] = A[i][j]`, rebuilt through a dense buffer so the result's rows carry
/// the same explicit-diagonal convention [`build_laplacian_csr`] emits.
pub fn permute_csr(csr: &LaplacianCsr, p: &[usize]) -> LaplacianCsr {
    let (row_ptrs, col_indices, values, n) = csr;
    let n = *n as usize;
    let mut dense = vec![0.0_f64; n * n];
    for row in 0..n {
        for k in row_ptrs[row] as usize..row_ptrs[row + 1] as usize {
            dense[p[row] * n + p[col_indices[k] as usize]] = values[k];
        }
    }
    let mut out_ptrs = Vec::with_capacity(n + 1);
    let (mut out_cols, mut out_vals) = (Vec::new(), Vec::new());
    out_ptrs.push(0u32);
    for row in 0..n {
        for col in 0..n {
            let value = dense[row * n + col];
            if row == col || value != 0.0 {
                out_cols.push(col as u32);
                out_vals.push(value);
            }
        }
        out_ptrs.push(out_cols.len() as u32);
    }
    (out_ptrs, out_cols, out_vals, n as u32)
}

pub fn permutation_strategy(n: usize) -> impl Strategy<Value = Vec<usize>> {
    Just((0..n).collect::<Vec<_>>()).prop_shuffle()
}

/// Components are the residue classes mod the returned `parts`, so they interleave and the
/// block-contiguous order is never the identity. Without this the suites reach a
/// non-identity `Permutation` in about 3 cases of 512, nearly always an involution.
pub fn interleaved_components_strategy() -> impl Strategy<Value = (LaplacianCsr, usize)> {
    (2usize..=3, 5usize..=9).prop_flat_map(|(parts, n)| {
        let pair_count = n * (n - 1) / 2;
        // Never zero within a class, so each class is a complete graph and the component
        // count is exactly `parts`.
        prop::collection::vec(1u8..=4, pair_count).prop_map(move |weights| {
            let mut edge_weights = vec![0u8; pair_count];
            let mut position = 0usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    if i % parts == j % parts {
                        edge_weights[position] = weights[position];
                    }
                    position += 1;
                }
            }
            (build_laplacian_csr(n, &edge_weights), parts)
        })
    })
}

/// Components as in [`interleaved_components_strategy`], but only class `0` carries
/// diagonal surplus, so the ground vertex attaches to that class alone and the
/// augmented graph stays disconnected.
pub fn one_grounded_component_strategy() -> impl Strategy<Value = (LaplacianCsr, usize)> {
    (2usize..=3, 5usize..=9).prop_flat_map(|(parts, n)| {
        let pair_count = n * (n - 1) / 2;
        (
            prop::collection::vec(1u8..=4, pair_count),
            prop::collection::vec(1u8..=5, n),
        )
            .prop_map(move |(weights, surpluses)| {
                let mut edge_weights = vec![0u8; pair_count];
                let mut position = 0usize;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if i % parts == j % parts {
                            edge_weights[position] = weights[position];
                        }
                        position += 1;
                    }
                }
                let (rp, ci, mut vals, n_u32) = build_laplacian_csr(n, &edge_weights);
                for i in (0..n).filter(|i| i % parts == 0) {
                    for k in rp[i] as usize..rp[i + 1] as usize {
                        if ci[k] as usize == i {
                            vals[k] += surpluses[i] as f64;
                        }
                    }
                }
                ((rp, ci, vals, n_u32), parts)
            })
    })
}

/// Zero-sum within each residue class, so every floating block's right-hand side is
/// consistent and the whole system has an exact solution to measure against.
pub fn per_component_consistent_rhs(n: usize, parts: usize) -> Vec<f64> {
    let mut rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7).sin() * 3.0).collect();
    for class in 1..parts {
        let members: Vec<usize> = (0..n).filter(|i| i % parts == class).collect();
        let mean = members.iter().map(|&i| rhs[i]).sum::<f64>() / members.len() as f64;
        for &i in &members {
            rhs[i] -= mean;
        }
    }
    rhs
}

pub fn laplacian_csr_strategy() -> impl Strategy<Value = LaplacianCsr> {
    (1usize..=8).prop_flat_map(|n| {
        let pair_count = n * (n - 1) / 2;
        prop::collection::vec(0u8..=4, pair_count)
            .prop_map(move |edge_weights| build_laplacian_csr(n, &edge_weights))
    })
}

pub fn rhs_for_dimension(n: usize) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n];
    if n >= 2 {
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
    }
    rhs
}

pub fn is_connected(row_ptrs: &[u32], col_indices: &[u32], n: u32) -> bool {
    if n <= 1 {
        return true;
    }
    let n = n as usize;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);
    while let Some(v) = queue.pop_front() {
        let start = row_ptrs[v] as usize;
        let end = row_ptrs[v + 1] as usize;
        for &col in &col_indices[start..end] {
            let u = col as usize;
            if !visited[u] {
                visited[u] = true;
                queue.push_back(u);
            }
        }
    }
    visited.iter().all(|&v| v)
}

pub fn random_zero_sum_rhs_strategy(n: usize) -> BoxedStrategy<Vec<f64>> {
    if n <= 1 {
        Just(vec![0.0; n]).boxed()
    } else {
        prop::collection::vec(-10.0f64..10.0, n)
            .prop_map(|mut v| {
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                for x in &mut v {
                    *x -= mean;
                }
                v
            })
            .boxed()
    }
}

pub fn laplacian_with_rhs_strategy() -> impl Strategy<Value = (LaplacianCsr, Vec<f64>)> {
    laplacian_csr_strategy().prop_flat_map(|(rp, ci, vals, n)| {
        random_zero_sum_rhs_strategy(n as usize)
            .prop_map(move |rhs| ((rp.clone(), ci.clone(), vals.clone(), n), rhs))
    })
}

pub fn sddm_csr_strategy() -> impl Strategy<Value = LaplacianCsr> {
    (1usize..=8).prop_flat_map(|n| {
        let pair_count = n * (n - 1) / 2;
        (
            prop::collection::vec(0u8..=4, pair_count),
            prop::collection::vec(1u8..=5, n),
        )
            .prop_map(move |(edge_weights, surpluses)| {
                let (rp, ci, mut vals, n_u32) = build_laplacian_csr(n, &edge_weights);
                for i in 0..n {
                    let start = rp[i] as usize;
                    let end = rp[i + 1] as usize;
                    for k in start..end {
                        if ci[k] as usize == i {
                            vals[k] += surpluses[i] as f64;
                        }
                    }
                }
                (rp, ci, vals, n_u32)
            })
    })
}
