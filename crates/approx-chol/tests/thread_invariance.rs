use approx_chol::input::{BipartiteGram, FactorWidth1, Weights};
use approx_chol::{factorize_with, Config, InnerParallelism, SchurMode};

const DEGREE: usize = 48;

/// `(left, right, frustrated)` per block: the first passes the `Cᵀ` gate, each splits its star fold and exact rows, together they split the components.
const GROUPS: [(usize, usize, bool); 3] =
    [(2800, 700, false), (420, 1000, true), (420, 1000, false)];

struct Gram {
    left: Vec<u32>,
    right: Vec<u32>,
    left_values: Vec<f64>,
    right_values: Vec<f64>,
}

fn gram() -> Gram {
    let mut g = Gram {
        left: Vec::new(),
        right: Vec::new(),
        left_values: Vec::new(),
        right_values: Vec::new(),
    };
    let (mut first_left, mut first_right) = (0, 0);
    for (left, right, frustrated) in GROUPS {
        for i in 0..left {
            for d in 0..DEGREE {
                let k = g.left.len();
                g.left.push((first_left + i) as u32);
                g.right
                    .push((first_right + (i * 7 + d * 13) % right) as u32);
                g.left_values.push(1.0 + (k % 7) as f64 / 8.0);
                let magnitude = 0.5 + (k % 3) as f64 / 4.0;
                let sign = if frustrated && k % 5 == 0 { 1.0 } else { -1.0 };
                g.right_values.push(sign * magnitude);
            }
        }
        (first_left, first_right) = (first_left + left, first_right + right);
    }
    g
}

/// `Factor::solve`, then each block's `solve_local` under `Allow` and `Forbid`, as bits.
fn solve_bits(g: &Gram, config: Config) -> Vec<u64> {
    let n_left = GROUPS.iter().map(|g| g.0).sum();
    let n_right = GROUPS.iter().map(|g| g.1).sum();
    let gram = BipartiteGram::new(
        FactorWidth1::new(n_left, &g.left, Some(&g.left_values)).unwrap(),
        FactorWidth1::new(n_right, &g.right, Some(&g.right_values)).unwrap(),
        Weights::unit(),
    )
    .unwrap();
    let factor = factorize_with(gram, config).unwrap();
    assert_eq!(factor.blocks().len(), GROUPS.len());
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
fn factors_and_solves_are_bit_identical_across_thread_counts() {
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
    for (label, config) in configs {
        let at = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| solve_bits(&g, config))
        };
        let serial = at(1);
        for threads in [2, 7] {
            assert!(
                at(threads) == serial,
                "{label}: {threads} threads differ from 1"
            );
        }
    }
}
