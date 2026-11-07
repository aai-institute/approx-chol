use approx_chol::{factorize_complement, factorize_with, Config, ScalingPolicy, SchurMode, Side};
use rstest::rstest;

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::solve::Solve;
use crate::common::threads;

const DEGREE: usize = 48;

/// `(left, right, frustrated)` per block: the first's 3200·48 entries clear the `Cᵀ` gate (2¹⁷) by 17%, as `factor.rs`'s `decoding_rebuilds_the_transpose` checks; each splits its star fold and exact rows, together they split the components.
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

/// The solve of `b` as bits, which its solve and each column of a four-column apply match under each budget; `kept` factors the complement.
fn solve_bits(g: &Bipartite, config: Config, kept: Option<Side>) -> Vec<u64> {
    let gram = g.view();
    let factor = match kept {
        None => factorize_with(gram, config, ScalingPolicy::default()).unwrap(),
        Some(side) => factorize_complement(gram, side, config, ScalingPolicy::default()).unwrap(),
    };
    assert_eq!(factor.components().len(), GROUPS.len());
    let b: Vec<f64> = (0..factor.n())
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let whole = bits(&factor.solve(&b).unwrap());
    for threads in threads::BUDGETS {
        assert_eq!(
            bits(&factor.solve_at(&b, threads).unwrap()),
            whole,
            "{threads:?}"
        );
    }
    let scale = [1.0, -0.5, 0.0, 3.0];
    let alone = scale.map(|s| {
        bits(
            &factor
                .solve(&b.iter().map(|&v| s * v).collect::<Vec<_>>())
                .unwrap(),
        )
    });
    for threads in threads::BUDGETS {
        let mut rows: Vec<[f64; 4]> = b.iter().map(|&v| scale.map(|s| s * v)).collect();
        factor.apply(&mut rows, threads).unwrap();
        for (j, alone) in alone.iter().enumerate() {
            let column: Vec<f64> = rows.iter().map(|row| row[j]).collect();
            assert_eq!(&bits(&column), alone, "column {j}, {threads:?}");
        }
    }
    whole
}

#[rstest]
#[case::ac(Config::default())]
#[case::ac2(Config { split_merge: Some(3), ..Config::default() })]
#[case::exact(Config { complement: SchurMode::Exact, ..Config::default() })]
fn solves_are_bit_identical_across_thread_counts(
    #[case] config: Config,
    #[values(None, Some(Side::Left), Some(Side::Right))] kept: Option<Side>,
) {
    let g = gram();
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
            "{config:?} {kept:?}: {threads} threads differ from 1"
        );
    }
}
