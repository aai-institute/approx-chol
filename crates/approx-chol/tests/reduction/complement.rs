use super::{dense, dense_residual, gram};
use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use approx_chol::{
    factorize_complement, factorize_with, Backend, Config, Factor, ScalingPolicy, Side,
};

/// A bipartite `V`'s columns `(left, right, x, y)`; no `x` and `y` reads as an incidence.
struct Bipartite {
    n_left: usize,
    n_right: usize,
    left: Vec<u32>,
    right: Vec<u32>,
    values: Option<(Vec<f64>, Vec<f64>)>,
}

impl Bipartite {
    /// `columns` columns over sides of `n_left` and `n_right`, left vertex `n_left − 1` in none, so it is dead.
    fn new(n_left: usize, n_right: usize, columns: usize, signed: bool) -> Self {
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize
        };
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

    fn view(&self) -> BipartiteFactorWidth2<'_, f64> {
        let (x, y) = match &self.values {
            Some((x, y)) => (Some(&x[..]), Some(&y[..])),
            None => (None, None),
        };
        BipartiteFactorWidth2::new(
            FactorWidth1::new(self.n_left, &self.left, x).expect("valid left"),
            FactorWidth1::new(self.n_right, &self.right, y).expect("valid right"),
        )
        .expect("valid gram")
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

    /// Each side's dense complement factor, left then right, checked to solve its complement.
    fn solves_dense(&self) -> [Factor; 2] {
        [Side::Left, Side::Right].map(|kept| {
            let factor =
                factorize_complement(self.view(), kept, dense::dense(), Default::default())
                    .expect("factors");
            dense_residual::solve_residual(&factor, &self.complement(kept));
            factor
        })
    }
}

fn max(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn an_exact_complement_solves_the_dense_complement() {
    for signed in [false, true] {
        Bipartite::new(31, 12, 60, signed).solves_dense();
    }
}

/// Right vertex 11 is in one column, at `x = 0`, so it is lone; kept, it is numbered by its side.
#[test]
fn a_kept_lone_vertex_is_numbered_by_its_side() {
    let mut gram = Bipartite::new(31, 12, 60, true);
    gram.right.iter_mut().for_each(|r| *r %= 11);
    gram.left.push(0);
    gram.right.push(11);
    let (x, y) = gram.values.as_mut().expect("signed");
    x.push(0.0);
    y.push(1.0);
    gram.solves_dense();
}

/// Keeping the side [`factorize_with`] keeps, the complement draws as the whole factor's Cholesky does.
#[test]
fn a_sampled_complement_is_the_whole_factors_kept_part() {
    let config = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };
    for signed in [false, true] {
        for (n_left, n_right, kept) in [(401, 60, Side::Right), (60, 401, Side::Left)] {
            let gram = Bipartite::new(n_left, n_right, 3000, signed);
            let s = gram.complement(kept);
            let whole =
                factorize_with(gram.view(), config, ScalingPolicy::default()).expect("factors");
            let factor = factorize_complement(gram.view(), kept, config, ScalingPolicy::default())
                .expect("factors");
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
            let delta: Vec<f64> = x
                .iter()
                .zip(&through_whole[kept_range])
                .map(|(a, b)| a - b)
                .collect();
            let gap = max(&dense_residual::apply(&s, &delta));
            assert!(gap <= 1e-10 * max(&b), "signed {signed} {kept:?}: {gap}");
        }
    }
}

#[test]
fn a_signed_trees_complement_keeps_its_sides_signs() {
    for (n_left, n_right) in [(6usize, 5usize), (5, 6)] {
        let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for k in 0..(n_left + n_right - 1) {
            left.push((k.div_ceil(2)).min(n_left - 1) as u32);
            right.push((k / 2).min(n_right - 1) as u32);
            x.push(if k % 3 == 1 { -1.0 } else { 1.0 });
            y.push(if k % 4 == 2 { -1.0 } else { 1.0 });
        }
        let gram = Bipartite {
            n_left,
            n_right,
            left,
            right,
            values: Some((x, y)),
        };
        for factor in gram.solves_dense() {
            assert_eq!(factor.blocks().len(), 1, "{n_left}x{n_right}");
        }
    }
}

#[test]
fn a_two_block_complement_numbers_each_block_by_the_kept_side() {
    for signed in [false, true] {
        let left = vec![0, 1, 1, 3, 4, 4, 5];
        let right = vec![0, 0, 1, 3, 3, 4, 5];
        let values = signed.then(|| {
            let x = vec![1.0, 0.7, -1.2, 0.9, 1.1, -0.8, 1.3];
            let y = vec![0.6, -1.0, 1.4, 0.8, -1.5, 1.0, 0.9];
            (x, y)
        });
        let gram = Bipartite {
            n_left: 6,
            n_right: 6,
            left,
            right,
            values,
        };
        for factor in gram.solves_dense() {
            assert!(factor.blocks().len() >= 2, "signed {signed}");
        }
    }
}
