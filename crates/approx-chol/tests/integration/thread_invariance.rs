use approx_chol::{factorize_complement, factorize_with, Config, ScalingPolicy, SchurMode, Side};
use rstest::rstest;

use crate::common::bipartite::Bipartite;
use crate::common::bits::bits;
use crate::common::solve::Solve;
use crate::common::threads;

/// The solve of `b` as bits, which its solve and each column of a four-column apply match under each budget; `kept` factors the complement.
fn solve_bits(g: &Bipartite, config: Config, kept: Option<Side>) -> Vec<u64> {
    let gram = g.view();
    let factor = match kept {
        None => factorize_with(gram, config, ScalingPolicy::default()).unwrap(),
        Some(side) => factorize_complement(gram, side, config, ScalingPolicy::default()).unwrap(),
    };
    assert_eq!(factor.components().len(), 3);
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
    let g = Bipartite::three_groups();
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
