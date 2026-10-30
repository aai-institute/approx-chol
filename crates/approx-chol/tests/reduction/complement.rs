use super::bipartite::Bipartite;
use super::lcg::Lcg;
use super::{dense, dense_residual, gram};
use approx_chol::{factorize_complement, factorize_with, Config, ScalingPolicy, Side};
use rstest::rstest;

impl Bipartite {
    /// `columns` columns over sides of `n_left` and `n_right`, left vertex `n_left − 1` in none, so it is dead.
    fn new(n_left: usize, n_right: usize, columns: usize, signed: bool) -> Self {
        let mut lcg = Lcg(0x9e37_79b9);
        let mut next = || (lcg.next_u64() >> 33) as usize;
        let left: Vec<u32> = (0..columns).map(|k| (k % (n_left - 1)) as u32).collect();
        let right: Vec<u32> = (0..columns).map(|_| (next() % n_right) as u32).collect();
        let values = signed.then(|| {
            let mut magnitude = || 0.5 + (next() % 1000) as f64 / 1000.0;
            let x: Vec<f64> = (0..columns).map(|_| magnitude()).collect();
            let y: Vec<f64> = (0..columns)
                .map(|k| magnitude() * if k % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            (x, y)
        });
        Self {
            n_left,
            n_right,
            left,
            right,
            values,
        }
    }

    /// `A`'s Schur complement onto `kept`, dense: the other side's block of a bipartite `A` is diagonal.
    fn complement(&self, kept: Side) -> Vec<Vec<f64>> {
        let ones = vec![1.0; self.left.len()];
        let (x, y) = self.values.as_ref().map_or((&ones, &ones), |(x, y)| (x, y));
        let shifted: Vec<u32> = self.right.iter().map(|&r| r + self.n_left as u32).collect();
        let a = gram::matrix(self.n_left + self.n_right, &self.left, &shifted, x, y);
        let (kept, eliminated) = match kept {
            Side::Left => (0..self.n_left, self.n_left..a.len()),
            Side::Right => (self.n_left..a.len(), 0..self.n_left),
        };
        let s_entry = |i: usize, j: usize| {
            let through = (eliminated.clone()).filter(|&e| a[e][e] != 0.0);
            a[i][j] - through.map(|e| a[i][e] * a[e][j] / a[e][e]).sum::<f64>()
        };
        (kept.clone())
            .map(|i| kept.clone().map(|j| s_entry(i, j)).collect())
            .collect()
    }

    /// Right vertex `n_right − 1` in one column, at `x = 0`, so it is lone.
    fn with_lone_vertex(mut self) -> Self {
        let lone = self.n_right as u32 - 1;
        self.right.iter_mut().for_each(|r| *r %= lone);
        self.left.push(0);
        self.right.push(lone);
        let (x, y) = self.values.as_mut().expect("signed");
        x.push(0.0);
        y.push(1.0);
        self
    }

    /// A spanning tree with unit-magnitude signs, so its congruence is `Signs`.
    fn signed_tree(n_left: usize, n_right: usize) -> Self {
        let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for k in 0..(n_left + n_right - 1) {
            left.push((k.div_ceil(2)).min(n_left - 1) as u32);
            right.push((k / 2).min(n_right - 1) as u32);
            x.push(if k % 3 == 1 { -1.0 } else { 1.0 });
            y.push(if k % 4 == 2 { -1.0 } else { 1.0 });
        }
        Self {
            n_left,
            n_right,
            left,
            right,
            values: Some((x, y)),
        }
    }

    /// Three components beside dead vertex `2` on each side, so each side's complement has three blocks.
    fn three_blocks(signed: bool) -> Self {
        let values = signed.then(|| {
            let x = vec![1.0, 0.7, -1.2, 0.9, 1.1, -0.8, 1.3];
            let y = vec![0.6, -1.0, 1.4, 0.8, -1.5, 1.0, 0.9];
            (x, y)
        });
        Self {
            n_left: 6,
            n_right: 6,
            left: vec![0, 1, 1, 3, 4, 4, 5],
            right: vec![0, 0, 1, 3, 3, 4, 5],
            values,
        }
    }
}

fn max(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[rstest]
#[case::lone_vertex(Bipartite::new(31, 12, 60, true).with_lone_vertex(), [1, 2])]
#[case::signed_tree(Bipartite::signed_tree(5, 6), [1, 1])]
#[case::three_blocks(Bipartite::three_blocks(false), [3, 3])]
fn an_exact_complement_solves_the_dense_complement(
    #[case] gram: Bipartite,
    #[case] blocks: [usize; 2],
) {
    for (kept, blocks) in [Side::Left, Side::Right].into_iter().zip(blocks) {
        let factor = factorize_complement(gram.view(), kept, dense::dense(), Default::default())
            .expect("factors");
        assert_eq!(factor.blocks().len(), blocks, "{kept:?}");
        dense_residual::solve_residual(&factor, &gram.complement(kept));
    }
}

/// Keeping the side [`factorize_with`] keeps, the complement draws as the whole factor's Cholesky does.
#[test]
fn a_sampled_complement_is_the_whole_factors_kept_part() {
    let (n_left, n_right, kept) = (60, 401, Side::Left);
    let config = Config {
        exact_max_dim: 0,
        ..Config::default()
    };
    let gram = Bipartite::new(n_left, n_right, 3000, false);
    let s = gram.complement(kept);
    let whole = factorize_with(gram.view(), config, ScalingPolicy::default()).expect("factors");
    let factor =
        factorize_complement(gram.view(), kept, config, ScalingPolicy::default()).expect("factors");
    let u: Vec<f64> = (0..s.len()).map(|i| ((5 * i + 2) as f64).cos()).collect();
    // In `S`'s range, so the padded `b` is in `A`'s.
    let b = dense_residual::apply(&s, &u);
    let kept_range = match kept {
        Side::Left => 0..n_left,
        Side::Right => n_left..n_left + n_right,
    };
    let mut padded = vec![0.0; n_left + n_right];
    padded[kept_range.clone()].copy_from_slice(&b);
    let through_whole = whole.solve(&padded).expect("b has length n");
    let x = factor.solve(&b).expect("b has length n");
    let delta: Vec<f64> = (x.iter())
        .zip(&through_whole[kept_range])
        .map(|(a, b)| a - b)
        .collect();
    let gap = max(&dense_residual::apply(&s, &delta));
    assert!(gap <= 1e-10 * max(&b), "{gap}");
}
