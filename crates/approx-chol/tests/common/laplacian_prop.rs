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

pub fn laplacian_csr_strategy() -> impl Strategy<Value = LaplacianCsr> {
    (1usize..=8).prop_flat_map(|n| {
        let pair_count = n * (n - 1) / 2;
        prop::collection::vec(0u8..=4, pair_count)
            .prop_map(move |edge_weights| build_laplacian_csr(n, &edge_weights))
    })
}

#[allow(dead_code)]
pub fn rhs_for_dimension(n: usize) -> Vec<f64> {
    let mut rhs = vec![0.0_f64; n];
    if n >= 2 {
        rhs[0] = 1.0;
        rhs[n - 1] = -1.0;
    }
    rhs
}

#[allow(dead_code)]
pub fn csr_matvec(row_ptrs: &[u32], col_indices: &[u32], values: &[f64], x: &[f64]) -> Vec<f64> {
    let n = row_ptrs.len() - 1;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for k in start..end {
            y[i] += values[k] * x[col_indices[k] as usize];
        }
    }
    y
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn laplacian_with_rhs_strategy() -> impl Strategy<Value = (LaplacianCsr, Vec<f64>)> {
    laplacian_csr_strategy().prop_flat_map(|(rp, ci, vals, n)| {
        random_zero_sum_rhs_strategy(n as usize)
            .prop_map(move |rhs| ((rp.clone(), ci.clone(), vals.clone(), n), rhs))
    })
}

#[allow(dead_code)]
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
