use proptest::prelude::*;

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
