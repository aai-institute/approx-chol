use approx_chol::input::Side;
use approx_chol::{
    factorize_complement, factorize_with, ApproxInverse, Config, Real, ScalingPolicy,
};

use super::bipartite::Bipartite;
use super::bits::bits;
use super::solve::Solve;

impl Bipartite {
    /// 200 left vertices of degree 6 on 80 right ones, every fifth entry of the opposite sign; `|x| ≠ |y|` leaves every row a surplus far past the band.
    pub fn frustrated() -> Self {
        let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for i in 0..200 {
            for d in 0..6 {
                let k = left.len();
                left.push(i as u32);
                right.push(((i * 7 + d * 13) % 80) as u32);
                x.push(1.0 + (k % 7) as f64 / 8.0);
                y.push(if k % 5 == 0 {
                    0.75
                } else {
                    -0.5 - (k % 3) as f64 / 4.0
                });
            }
        }
        Self {
            n_left: 200,
            n_right: 80,
            left,
            right,
            values: Some((x, y)),
        }
    }
}

/// `g` factored under `config`, whole and onto each side, by name.
pub fn factors(g: &Bipartite, config: Config) -> Vec<(String, ApproxInverse)> {
    let policy = ScalingPolicy::default();
    let mut factors = vec![(
        "whole".to_string(),
        factorize_with(g.view(), config, policy).unwrap(),
    )];
    for side in [Side::Left, Side::Right] {
        let onto = factorize_complement(g.view(), side, config, policy).unwrap();
        factors.push((format!("onto {side:?}"), onto));
    }
    factors
}

/// Whether a frustrated block of `factor` is eliminated signed.
pub fn is_signed<T: Real>(factor: &ApproxInverse<T>) -> bool {
    format!("{factor:?}").contains("Signed(")
}

/// The solve of a fixed `b` under `config`, whole and onto each side, as bits.
pub fn solves(g: &Bipartite, config: Config) -> Vec<Vec<u64>> {
    (factors(g, config).iter())
        .map(|(_, factor)| {
            let b: Vec<f64> = (0..factor.n()).map(|i| (i % 7) as f64 - 3.0).collect();
            bits(&factor.solve(&b).unwrap())
        })
        .collect()
}
