#![cfg(feature = "serde")]

use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1, Laplacian, Sddm, Weights};
use approx_chol::{
    factorize_with, Backend, BlockFactor, Config, Factor, InnerParallelism, ScalingPolicy,
    SchurMode,
};
use rstest::rstest;

fn factors(config: Config) -> Vec<Factor> {
    let (left, right, per_row) = (12_000u32, 400u32, 12u32);
    let rows: Vec<u32> = (0..left * per_row).map(|k| k / per_row).collect();
    let cols: Vec<u32> = (0..left * per_row)
        .map(|k| (k * 7 + k / per_row) % right)
        .collect();
    let slope: Vec<f64> = (0..rows.len())
        .map(|k| 1.0 + (k % 5) as f64 / 4.0)
        .collect();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(left as usize, &rows, Some(&slope)).unwrap(),
        FactorWidth1::new(right as usize, &cols, None).unwrap(),
        Weights::Unit,
    )
    .unwrap();
    let bipartite = factorize_with(gram, config).unwrap();

    let (a, b): (Vec<u32>, Vec<u32>) = (0..60u32)
        .filter(|i| i % 20 != 19)
        .map(|i| (i, i + 1))
        .unzip();
    let ground: Vec<f64> = (0..61).map(|i| if i < 20 { 0.0 } else { 0.5 }).collect();
    let laplacian = Sddm::new(Laplacian::new(61, &a, &b).unwrap(), &ground).unwrap();
    let laplacian = factorize_with(laplacian, config).unwrap();

    let (c, d): (Vec<u32>, Vec<u32>) = (0..58u32).map(|i| (i, i + 1 + i % 2)).unzip();
    let positive: Vec<bool> = (0..c.len()).map(|k| k % 3 == 0).collect();
    let mut dense = vec![vec![0.0; 60]; 60];
    for k in 0..c.len() {
        let (i, j, cell) = (
            c[k] as usize,
            d[k] as usize,
            if positive[k] { -1.0 } else { 1.0 },
        );
        (dense[i][i], dense[j][j], dense[i][j], dense[j][i]) =
            (dense[i][i] + 1.0, dense[j][j] + 1.0, cell, cell);
    }
    let (mut ptrs, mut columns, mut values) = (vec![0u32], vec![], vec![]);
    for row in &dense {
        for (j, &x) in row.iter().enumerate().filter(|&(_, &x)| x != 0.0) {
            columns.push(j as u32);
            values.push(x);
        }
        ptrs.push(columns.len() as u32);
    }
    let signed = CsrRef::new(&ptrs, &columns, &values, 60).unwrap();
    let signed = factorize_with(signed, config).unwrap();
    vec![bipartite, laplacian, signed]
}

fn configs() -> [Config; 2] {
    let exact = Config {
        complement: SchurMode::Exact,
        backend: Backend::Approximate,
        ..Config::default()
    };
    [Config::default(), exact]
}

fn solve(block: &BlockFactor, inner: InnerParallelism) -> Vec<u64> {
    let mut x: Vec<f64> = (0..block.vertices().len())
        .map(|i| (i % 7) as f64 - 3.0)
        .collect();
    let mut scratch = vec![0.0; block.scratch_len()];
    block.solve_local(&mut x, &mut scratch, inner).unwrap();
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn round_trip_solves_bit_identically() {
    let defaults = factors(Config::default());
    assert!(defaults
        .iter()
        .flat_map(Factor::blocks)
        .any(|block| block.floats()));
    assert!(
        defaults[0]
            .blocks()
            .iter()
            .any(|block| block.splittable_work() > 0),
        "a block keeps Cᵀ, which loading rebuilds"
    );
    for config in configs() {
        for factor in factors(config) {
            let blocks = factor.into_blocks();
            let bytes = postcard::to_stdvec(&blocks).unwrap();
            let back: Vec<BlockFactor> = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(postcard::to_stdvec(&back).unwrap(), bytes);
            for (block, back) in blocks.iter().zip(&back) {
                for inner in [InnerParallelism::Allow, InnerParallelism::Forbid] {
                    assert_eq!(solve(block, inner), solve(back, inner));
                }
            }
        }
    }
}

#[test]
fn corrupt_bytes_error_instead_of_panicking() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as usize
    };
    for factor in factors(Config::default()) {
        let bytes = postcard::to_stdvec(&factor.into_blocks()).unwrap();
        for trial in 0..500 {
            let mut corrupt = bytes.clone();
            let at = next() % corrupt.len();
            match trial % 3 {
                0 => corrupt[at] = next() as u8,
                1 => corrupt[at] ^= 1 << (next() % 8),
                _ => corrupt.truncate(at),
            }
            if let Ok(blocks) = postcard::from_bytes::<Vec<BlockFactor>>(&corrupt) {
                blocks
                    .iter()
                    .for_each(|block| drop(solve(block, InnerParallelism::Forbid)));
            }
        }
    }
}

fn varint(mut x: u64, bytes: &mut Vec<u8>) {
    while x >= 0x80 {
        bytes.push(x as u8 | 0x80);
        x >>= 7;
    }
    bytes.push(x as u8);
}

/// One `u32` block on vertices `[64, 0]` whose elimination has `rows` empty rows of `C`, cut off before its Cholesky.
fn truncated_elimination(rows: u64, n_cols: u64) -> Vec<u8> {
    let mut bytes = vec![1];
    varint(approx_chol::FACTOR_FORMAT_VERSION.into(), &mut bytes);
    bytes.extend([0, 2, 64, 0, 0, 1]);
    varint(rows + 1, &mut bytes);
    bytes.extend(std::iter::repeat_n(0, rows as usize + 1));
    bytes.extend([0, 0]);
    varint(n_cols, &mut bytes);
    varint(rows, &mut bytes);
    (0..rows).for_each(|_| bytes.extend(1.0f64.to_le_bytes()));
    bytes
}

#[rstest]
#[case::dimension_overflows(3, u64::MAX)]
#[case::transpose_overflows(10_001, 1 << 62)]
#[case::one_more_kept(3, 4)]
fn an_elimination_keeping_more_columns_than_rows_is_corrupt(
    #[case] rows: u64,
    #[case] n_cols: u64,
) {
    let error = postcard::from_bytes::<Vec<BlockFactor>>(&truncated_elimination(rows, n_cols));
    assert_eq!(error.unwrap_err(), postcard::Error::SerdeDeCustom);
}

#[rstest]
#[case::kept_in_place(3)]
#[case::transposed(10_001)]
fn an_elimination_keeping_at_most_its_rows_decodes_up_to_the_cut(#[case] rows: u64) {
    let error = postcard::from_bytes::<Vec<BlockFactor>>(&truncated_elimination(rows, rows));
    assert_eq!(
        error.unwrap_err(),
        postcard::Error::DeserializeUnexpectedEnd
    );
}

/// A Laplacian on `0..4` beside two zero vertices, so the factor carries dead vertices.
fn with_isolated_vertex() -> Factor {
    let edge = Laplacian::new(3, &[0u32], &[1]).unwrap();
    factorize_with(
        Sddm::new(edge, &[1.0, 0.0, 2.0]).unwrap(),
        Config::default(),
    )
    .unwrap()
}

fn with_dead_vertices() -> Factor {
    let laplacian = Laplacian::new(6, &[0u32, 1, 2], &[1u32, 2, 3]).unwrap();
    factorize_with(laplacian, Config::default()).unwrap()
}

fn solve_factor(factor: &Factor) -> Vec<u64> {
    let b: Vec<f64> = (0..factor.n()).map(|i| (i % 5) as f64 - 2.0).collect();
    let x = factor.solve(&b).unwrap();
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn a_factor_round_trips_bit_identically() {
    let all = configs()
        .into_iter()
        .flat_map(factors)
        .chain([with_dead_vertices()]);
    for factor in all {
        let bytes = postcard::to_stdvec(&factor).unwrap();
        let version = postcard::to_stdvec(&approx_chol::FACTOR_FORMAT_VERSION).unwrap();
        assert!(bytes.starts_with(&version), "the version leads");
        let decoded: Factor = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(
            (decoded.n(), decoded.n_steps()),
            (factor.n(), factor.n_steps())
        );
        assert_eq!(solve_factor(&decoded), solve_factor(&factor));
    }
}

/// The factor's bytes with `n` and the dead vertices replaced, its blocks kept.
fn respliced(factor: &Factor, n: usize, dead: &[usize]) -> Vec<u8> {
    let bytes = postcard::to_stdvec(factor).unwrap();
    let version = approx_chol::FACTOR_FORMAT_VERSION;
    let own_dead: Vec<usize> = (0..factor.n())
        .filter(|&v| {
            factor
                .blocks()
                .iter()
                .all(|block| block.vertices().all(|u| u != v))
        })
        .collect();
    let head = postcard::to_stdvec(&(version, factor.n(), &own_dead)).unwrap();
    assert_eq!(
        &bytes[..head.len()],
        &head[..],
        "the head is version, n, dead"
    );
    let mut spliced = postcard::to_stdvec(&(version, n, dead)).unwrap();
    spliced.extend_from_slice(&bytes[head.len()..]);
    spliced
}

#[rstest]
#[case::a_vertex_missing(6, &[4])]
#[case::one_past_n(7, &[4, 5])]
#[case::dead_inside_a_block(6, &[4, 3])]
#[case::dead_out_of_range(6, &[4, 6])]
fn a_factor_not_covering_n_once_is_rejected(#[case] n: usize, #[case] dead: &[usize]) {
    let factor = with_dead_vertices();
    assert!(postcard::from_bytes::<Factor>(&respliced(&factor, 6, &[4, 5])).is_ok());
    let error = postcard::from_bytes::<Factor>(&respliced(&factor, n, dead)).unwrap_err();
    assert_eq!(error, postcard::Error::SerdeDeCustom);
}

/// Each `Config` as 0.6.0 serialized it to JSON and to postcard, beside what it reads as now.
fn released_configs() -> [(&'static str, &'static [u8], Config); 4] {
    let exact_below = |max_dim| Backend::ExactBelow { max_dim };
    [
        (
            r#"{"seed":0,"split_merge":null,"backend":{"ExactBelow":{"max_dim":24,"on_failure":"FallBackToApproximate"}}}"#,
            &[0, 0, 1, 24, 0],
            Config::default(),
        ),
        (
            r#"{"seed":7,"split_merge":3,"backend":{"ExactBelow":{"max_dim":24,"on_failure":"FallBackToApproximate"}}}"#,
            &[7, 1, 3, 1, 24, 0],
            Config {
                seed: 7,
                split_merge: Some(3),
                ..Config::default()
            },
        ),
        (
            r#"{"seed":0,"split_merge":null,"backend":"Approximate"}"#,
            &[0, 0, 0],
            Config {
                backend: Backend::Approximate,
                ..Config::default()
            },
        ),
        (
            r#"{"seed":0,"split_merge":null,"backend":{"ExactBelow":{"max_dim":64,"on_failure":"Error"}}}"#,
            &[0, 0, 1, 64, 1],
            Config {
                backend: exact_below(64),
                ..Config::default()
            },
        ),
    ]
}

#[test]
fn a_released_json_config_reads_as_its_new_config() {
    for (json, _, expected) in released_configs() {
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config, expected, "{json}");
    }
}

// Postcard is not self-describing, so a 0.6.0 config must fail rather than misread its `on_failure` as `complement`.
#[test]
fn a_released_postcard_config_is_rejected() {
    for (json, bytes, _) in released_configs() {
        assert!(postcard::from_bytes::<Config>(bytes).is_err(), "{json}");
    }
}

#[test]
fn a_scaling_policy_decode_rejects_tolerance_outside_unit_interval() {
    for tolerance in [2.0f64, -0.5, 1.0, f64::NAN] {
        let bytes = postcard::to_stdvec(&(tolerance, 7usize)).unwrap();
        assert!(
            postcard::from_bytes::<ScalingPolicy>(&bytes).is_err(),
            "{tolerance}"
        );
    }
    let bytes = postcard::to_stdvec(&(0.25f64, 7usize)).unwrap();
    let policy: ScalingPolicy = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(policy, ScalingPolicy::new(0.25, 7).unwrap());
    assert_eq!(postcard::to_stdvec(&policy).unwrap(), bytes);
}

/// Applies `f` to the node at `pointer` below every node of `v`; the count of nodes it edited.
fn edit(
    v: &mut serde_json::Value,
    pointer: &str,
    f: &mut impl FnMut(&mut serde_json::Value),
) -> usize {
    let own = usize::from(v.pointer_mut(pointer).map(&mut *f).is_some());
    own + match v {
        serde_json::Value::Object(map) => map.values_mut().map(|v| edit(v, pointer, f)).sum(),
        serde_json::Value::Array(items) => items.iter_mut().map(|v| edit(v, pointer, f)).sum(),
        _ => 0,
    }
}

/// Decodes `factor` after scaling by `by` every multiplier (a stored value below one) in each array at `pointer` below any node, or the number there.
fn decode_scaled(factor: &Factor, pointer: &str, by: f64) -> Result<Factor, serde_json::Error> {
    let scale = |x: &mut serde_json::Value| *x = (x.as_f64().unwrap() * by).into();
    let mut v = serde_json::to_value(factor).unwrap();
    let edited = edit(&mut v, pointer, &mut |node| match node {
        serde_json::Value::Array(values) => (values.iter_mut())
            .filter(|x| x.as_f64().is_some_and(|f| f < 1.0))
            .for_each(scale),
        x => scale(x),
    });
    assert!(edited > 0, "the factor stores {pointer}");
    serde_json::from_value::<Factor>(v)
}

/// Each stored `|D_e⁻¹C|` row sums to at most one; past it the backward sum can overflow.
#[test]
fn an_elimination_whose_rows_sum_past_one_is_corrupt() {
    let rows: Vec<u32> = (0..300).map(|o| o % 60).collect();
    let cols: Vec<u32> = (0..300).map(|o| (o * 7 + o / 13) % 30).collect();
    let weights: Vec<f64> = (0..300).map(|o| 1.0 + (o % 5) as f64).collect();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(60, &rows, None).unwrap(),
        FactorWidth1::new(30, &cols, None).unwrap(),
        Weights::Of(&weights),
    )
    .unwrap();
    let factor = approx_chol::factorize(gram).unwrap();
    decode_scaled(&factor, "/cross/values", 1.0).expect("a stored elimination decodes");
    decode_scaled(&factor, "/cross/values", 2.0).expect_err("rows summing to two are corrupt");
}

/// Each column of a dense `|L|` and each step of an approximate one sums to at most one; past it the substitution can overflow.
#[rstest]
#[case::dense("/Dense", Backend::ExactBelow { max_dim: 24 })]
#[case::approximate("/coefficients", Backend::Approximate)]
fn a_cholesky_whose_multipliers_pass_one_is_corrupt(
    #[case] stored: &str,
    #[case] backend: Backend,
) {
    let (a, b): (Vec<u32>, Vec<u32>) = (0..16u32)
        .flat_map(|v| [(v, v + 1), (v, v + 4)])
        .filter(|&(v, w)| w < 16 && (w != v + 1 || w % 4 != 0))
        .unzip();
    let ground: Vec<f64> = (0..16).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let grid = Laplacian::new(16, &a, &b).unwrap();
    let grid = Sddm::new(grid, &ground).unwrap();
    let config = Config {
        backend,
        ..Config::default()
    };
    let factor = factorize_with(grid, config).unwrap();
    decode_scaled(&factor, stored, 1.0).expect("a stored Cholesky decodes");
    decode_scaled(&factor, stored, 2.0).expect_err("multipliers summing to two are corrupt");
}

/// An approximate step's pivot scale is positive, as every other stored pivot; a negated one solves indefinite.
#[rstest]
#[case::negative(-1.0)]
#[case::zero(0.0)]
fn an_approximate_pivot_scale_not_positive_is_corrupt(#[case] by: f64) {
    let (a, b): (Vec<u32>, Vec<u32>) = (0..29u32).map(|v| (v, v + 1)).unzip();
    let path = Laplacian::new(30, &a, &b).unwrap();
    let config = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };
    let factor: Factor = factorize_with(path, config).unwrap();
    decode_scaled(&factor, "/pivot_scale", by).expect_err("a pivot scale not positive is corrupt");
}

/// A factor holding an isolated vertex reads back from a self-describing format too.
#[test]
fn an_isolated_vertex_round_trips_through_json() {
    let factor = with_isolated_vertex();
    let json = serde_json::to_string(&factor).unwrap();
    let back: Factor = serde_json::from_str(&json).unwrap();
    let b = [1.0, 2.0, 3.0];
    assert_eq!(factor.solve(&b).unwrap(), back.solve(&b).unwrap());
}

/// A forged empty `F` is a decode error, never a panic while its peak is read.
#[test]
fn empty_scales_are_corrupt() {
    let csr = CsrRef::new(&[0u32, 2, 4], &[0u32, 1, 0, 1], &[1.0, -2.0, -2.0, 5.0], 2).unwrap();
    let mut v = serde_json::to_value(approx_chol::factorize(csr).unwrap()).unwrap();
    let mut empty = |s: &mut serde_json::Value| *s = serde_json::json!([]);
    assert!(
        edit(&mut v, "/Scaled", &mut empty) > 0,
        "the factor stores scales"
    );
    serde_json::from_value::<Factor>(v).expect_err("empty scales are corrupt");
}

/// A forged u64 step length past the stored neighbors is a decode error, never an overflow.
#[test]
fn an_approximate_step_length_past_its_neighbors_is_corrupt() {
    let mut bytes = Vec::new();
    varint(u64::from(approx_chol::FACTOR_FORMAT_VERSION), &mut bytes);
    bytes.push(1); // Storage::U64
    bytes.extend([4, 0, 1, 2, 3, 0]); // vertices, Identity
    bytes.push(0); // no elimination
    bytes.extend([0, 0, 4, 0, 1]); // One, floating, n = 4, pinned 0, Approximate
    bytes.push(3);
    for (vertex, len) in [(3, 1), (2, u64::MAX), (1, 1)] {
        varint(vertex, &mut bytes);
        varint(len, &mut bytes);
        bytes.extend(1.0f64.to_le_bytes());
    }
    bytes.extend([3, 2, 1, 0, 3]);
    (0..3).for_each(|_| bytes.extend(1.0f64.to_le_bytes()));
    bytes.push(0); // no report
    postcard::from_bytes::<BlockFactor>(&bytes).expect_err("a step past its neighbors is corrupt");
}

/// An isolated vertex's inverse pivot, `0.5` here, scaled outside what a stored pivot inverts to is corrupt.
#[rstest]
#[case::zero(0.0)]
#[case::negative(-1.0)]
#[case::subnormal(1e-323)]
#[case::past_any_pivot(1e308)]
fn an_isolated_inverse_pivot_out_of_range_is_corrupt(#[case] by: f64) {
    let factor = with_isolated_vertex();
    decode_scaled(&factor, "/Isolated/inverse_pivot", by)
        .expect_err("its inverse pivot is out of range");
}
