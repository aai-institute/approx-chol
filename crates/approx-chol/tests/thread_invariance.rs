#[path = "common/bipartite.rs"]
mod bipartite;

use approx_chol::{
    factorize_complement, factorize_with, Config, InnerParallelism, ScalingPolicy, SchurMode, Side,
};
use bipartite::Bipartite;

const DEGREE: usize = 48;

/// `(left, right, frustrated)` per block: the first's 3200·48 entries clear the `Cᵀ` gate (2¹⁷) by 17%, each splits its star fold and exact rows, together they split the components.
const GROUPS: [(usize, usize, bool); 3] =
    [(3200, 700, false), (420, 1000, true), (420, 1000, false)];

fn gram() -> Bipartite {
    let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
    let (mut first_left, mut first_right) = (0, 0);
    for (n_left, n_right, frustrated) in GROUPS {
        for i in 0..n_left {
            for d in 0..DEGREE {
                let k = left.len();
                left.push((first_left + i) as u32);
                right.push((first_right + (i * 7 + d * 13) % n_right) as u32);
                x.push(1.0 + (k % 7) as f64 / 8.0);
                let magnitude = 0.5 + (k % 3) as f64 / 4.0;
                let sign = if frustrated && k % 5 == 0 { 1.0 } else { -1.0 };
                y.push(sign * magnitude);
            }
        }
        (first_left, first_right) = (first_left + n_left, first_right + n_right);
    }
    Bipartite {
        n_left: first_left,
        n_right: first_right,
        left,
        right,
        values: Some((x, y)),
    }
}

/// `Factor::solve`, then each block's `solve_local` under `Allow` and `Forbid`, as bits; `kept` factors the complement.
fn solve_bits(g: &Bipartite, config: Config, kept: Option<Side>) -> Vec<u64> {
    let gram = g.view();
    let factor = match kept {
        None => factorize_with(gram, config, ScalingPolicy::default()).unwrap(),
        Some(side) => factorize_complement(gram, side, config, ScalingPolicy::default()).unwrap(),
    };
    assert_eq!(factor.blocks().len(), GROUPS.len());
    assert!(
        kept.is_some() || factor.blocks().iter().any(|b| b.splittable_work() > 0),
        "a block stores Cᵀ, so Allow splits its solve"
    );
    let b: Vec<f64> = (0..factor.n())
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let x = factor.solve(&b).unwrap();
    let mut bits: Vec<u64> = x.iter().map(|x| x.to_bits()).collect();
    for block in factor.blocks() {
        for inner in [InnerParallelism::Allow, InnerParallelism::Forbid] {
            let mut x: Vec<f64> = block.vertices().map(|v| b[v]).collect();
            let mut scratch = vec![0.0; block.scratch_len()];
            block.solve_local(&mut x, &mut scratch, inner).unwrap();
            bits.extend(x.iter().map(|x| x.to_bits()));
        }
    }
    bits
}

#[test]
fn solves_are_bit_identical_across_thread_counts() {
    let g = gram();
    let configs = [
        ("AC", Config::default()),
        (
            "AC2",
            Config {
                split_merge: Some(3),
                ..Config::default()
            },
        ),
        (
            "exact",
            Config {
                complement: SchurMode::Exact,
                ..Config::default()
            },
        ),
    ];
    for ((label, config), kept) in configs
        .into_iter()
        .flat_map(|c| [None, Some(Side::Left), Some(Side::Right)].map(|k| (c, k)))
    {
        let at = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| solve_bits(&g, config, kept))
        };
        let serial = at(1);
        for threads in [2, 7] {
            assert!(
                at(threads) == serial,
                "{label} {kept:?}: {threads} threads differ from 1"
            );
        }
    }
}
