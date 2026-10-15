use approx_chol_next::ScalingPolicy;
use rstest::rstest;

use super::{reduce, reduce_with};

/// `D L D` for the graph `edges` on `n` vertices, `D` within `spread` of `I`: GDD by construction, `R = D⁻¹`.
fn dld(
    n: usize,
    edges: &[(usize, usize)],
    spread: f64,
    seed: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    // Arithmetic only, no libm: the same bits on every platform.
    let d: Vec<f64> = (0..n)
        .map(|i| 1.0 + spread * (((7919 * i + 104_729 * seed) % 1000) as f64 / 1000.0 - 0.5))
        .collect();
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j) in edges {
        rows[i].push((j, -(d[i] * d[j])));
        rows[j].push((i, -(d[j] * d[i])));
    }
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for (i, row) in rows.iter_mut().enumerate() {
        row.push((i, d[i] * row.len() as f64 * d[i]));
        row.sort_by_key(|&(j, _)| j);
        for &(j, value) in row.iter() {
            columns.push(j as u32);
            values.push(value);
        }
        row_ptrs.push(columns.len() as u32);
    }
    (row_ptrs, columns, values)
}

fn path(n: usize) -> Vec<(usize, usize)> {
    (1..n).map(|i| (i - 1, i)).collect()
}

fn grid(k: usize) -> Vec<(usize, usize)> {
    let right = (0..k * k).filter(|v| v % k + 1 < k).map(|v| (v, v + 1));
    let down = (0..k * (k - 1)).map(|v| (v, v + k));
    right.chain(down).collect()
}

/// CG stalls past 4× the settle band at n = 5000; 6144 steps cover the path, which 2048 do not (#194).
#[rstest]
#[case::spread_1e_6(1e-6)]
#[case::spread_1e_3(1e-3)]
#[case::spread_1(1.0)]
fn a_settled_dld_path_is_not_reported(#[case] spread: f64) {
    let n = 5000;
    let (row_ptrs, columns, values) = dld(n, &path(n), spread, 0);
    let policy = ScalingPolicy::new(1e-9, 6144).expect("a valid policy");
    let sgdd = reduce_with(&row_ptrs, &columns, &values, policy).expect("admitted");
    let block = &sgdd.blocks()[0];
    assert_eq!(block.scaling_report(), None, "spread {spread:e}");
    assert!(block.floats(), "spread {spread:e}");
}

/// Settled within 16ε, then `F·A·F` rounds a few ε more: this grid's 17.3ε is rounding, not a deficit.
#[test]
fn a_settled_scaling_with_rounding_deficit_is_not_clamped() {
    let k = 22;
    let (row_ptrs, columns, values) = dld(k * k, &grid(k), 1e-3, 30);
    let sgdd = reduce(&row_ptrs, &columns, &values).expect("admitted");
    assert_eq!(sgdd.blocks()[0].scaling_report(), None);
}
