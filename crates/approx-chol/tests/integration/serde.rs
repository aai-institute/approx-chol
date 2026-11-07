#![cfg(feature = "serde")]

use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1, Laplacian, Sddm};
use approx_chol::{factorize_with, ApproxInverse, Config, ScalingPolicy, SchurMode, Threads};
use rstest::rstest;

use crate::common::bits::bits;
use crate::common::gram::{dense_csr, matrix};
use crate::common::threads;

fn factors(config: Config) -> Vec<ApproxInverse> {
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
    )
    .unwrap();
    let bipartite = factorize_with(gram, config, ScalingPolicy::default()).unwrap();

    let (a, b): (Vec<u32>, Vec<u32>) = (0..60u32)
        .filter(|i| i % 20 != 19)
        .map(|i| (i, i + 1))
        .unzip();
    let ground: Vec<f64> = (0..61).map(|i| if i < 20 { 0.0 } else { 0.5 }).collect();
    let laplacian = Sddm::new(Laplacian::new(61, &a, &b).unwrap(), &ground).unwrap();
    let laplacian = factorize_with(laplacian, config, ScalingPolicy::default()).unwrap();

    let (c, d): (Vec<u32>, Vec<u32>) = (0..58u32).map(|i| (i, i + 1 + i % 2)).unzip();
    let cell: Vec<f64> = (0..c.len())
        .map(|k| if k % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let (ptrs, columns, values) = dense_csr(&matrix(60, &c, &d, &vec![1.0; c.len()], &cell));
    let signed = CsrRef::new(&ptrs, &columns, &values, 60).unwrap();
    let signed = factorize_with(signed, config, ScalingPolicy::default()).unwrap();
    vec![bipartite, laplacian, signed]
}

fn configs() -> [Config; 2] {
    let exact = Config {
        complement: SchurMode::Exact,
        exact_max_dim: 0,
        ..Config::default()
    };
    [Config::default(), exact]
}

fn solve(factor: &ApproxInverse, threads: Threads) -> Vec<u64> {
    let mut x: Vec<f64> = (0..factor.n()).map(|i| (i % 7) as f64 - 3.0).collect();
    factor.apply(&mut x, threads).unwrap();
    bits(&x)
}

#[test]
fn round_trip_solves_bit_identically() {
    let defaults = factors(Config::default());
    assert!(defaults
        .iter()
        .flat_map(ApproxInverse::components)
        .any(|component| component.floats()));
    for config in configs() {
        for factor in factors(config) {
            let bytes = postcard::to_stdvec(&factor).unwrap();
            let back: ApproxInverse = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(postcard::to_stdvec(&back).unwrap(), bytes);
            for threads in threads::BUDGETS {
                assert_eq!(solve(&factor, threads), solve(&back, threads));
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
        let bytes = postcard::to_stdvec(&factor).unwrap();
        for trial in 0..500 {
            let mut corrupt = bytes.clone();
            let at = next() % corrupt.len();
            match trial % 3 {
                0 => corrupt[at] = next() as u8,
                1 => corrupt[at] ^= 1 << (next() % 8),
                _ => corrupt.truncate(at),
            }
            if let Ok(factor) = postcard::from_bytes::<ApproxInverse>(&corrupt) {
                drop(solve(&factor, threads::ONE));
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

/// The version and a factor's head before its one block: `n`, no dead vertex, one block.
fn one_block_factor(n: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    varint(approx_chol::FACTOR_FORMAT_VERSION.into(), &mut bytes);
    varint(n, &mut bytes);
    bytes.extend([0, 1]);
    bytes
}

/// A factor of one `u32` block on vertices `[64, 0]` whose elimination has `rows` empty rows of `C`, cut off before its Cholesky.
fn truncated_elimination(rows: u64, n_cols: u64) -> Vec<u8> {
    let mut bytes = one_block_factor(65);
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
    let error = postcard::from_bytes::<ApproxInverse>(&truncated_elimination(rows, n_cols));
    assert_eq!(error.unwrap_err(), postcard::Error::SerdeDeCustom);
}

#[rstest]
#[case::kept_in_place(3)]
#[case::transposed(10_001)]
fn an_elimination_keeping_at_most_its_rows_decodes_up_to_the_cut(#[case] rows: u64) {
    let error = postcard::from_bytes::<ApproxInverse>(&truncated_elimination(rows, rows));
    assert_eq!(
        error.unwrap_err(),
        postcard::Error::DeserializeUnexpectedEnd
    );
}

/// A Laplacian on `0..4` beside two zero vertices, so the factor carries dead vertices.
fn with_isolated_vertex() -> ApproxInverse {
    let edge = Laplacian::new(3, &[0u32], &[1]).unwrap();
    factorize_with(
        Sddm::new(edge, &[1.0, 0.0, 2.0]).unwrap(),
        Config::default(),
        ScalingPolicy::default(),
    )
    .unwrap()
}

fn with_dead_vertices() -> ApproxInverse {
    let laplacian = Laplacian::new(6, &[0u32, 1, 2], &[1u32, 2, 3]).unwrap();
    factorize_with(laplacian, Config::default(), ScalingPolicy::default()).unwrap()
}

fn solve_factor(factor: &ApproxInverse) -> Vec<u64> {
    let b: Vec<f64> = (0..factor.n()).map(|i| (i % 5) as f64 - 2.0).collect();
    let x = factor.solve(&b).unwrap();
    bits(&x)
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
        let decoded: ApproxInverse = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(
            (decoded.n(), decoded.n_steps()),
            (factor.n(), factor.n_steps())
        );
        assert_eq!(solve_factor(&decoded), solve_factor(&factor));
    }
}

/// The factor's bytes with `n` and the dead vertices replaced, its blocks kept.
fn respliced(factor: &ApproxInverse, n: usize, dead: &[usize]) -> Vec<u8> {
    let bytes = postcard::to_stdvec(factor).unwrap();
    let version = approx_chol::FACTOR_FORMAT_VERSION;
    let own_dead: Vec<usize> = (0..factor.n())
        .filter(|&v| {
            factor
                .components()
                .all(|component| component.vertices().all(|u| u != v))
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
    assert!(postcard::from_bytes::<ApproxInverse>(&respliced(&factor, 6, &[4, 5])).is_ok());
    let error = postcard::from_bytes::<ApproxInverse>(&respliced(&factor, n, dead)).unwrap_err();
    assert_eq!(error, postcard::Error::SerdeDeCustom);
}

/// 0.6.0's JSON lacks `exact_max_dim` and `complement`, so it is rejected, not read as the default bound.
#[rstest]
#[case::default(r#"{"seed":0,"split_merge":null,"backend":{"ExactBelow":{"max_dim":24,"on_failure":"FallBackToApproximate"}}}"#)]
#[case::approximate(r#"{"seed":0,"split_merge":null,"backend":"Approximate"}"#)]
fn a_released_json_config_is_rejected(#[case] json: &str) {
    assert!(serde_json::from_str::<Config>(json).is_err());
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
fn decode_scaled(
    factor: &ApproxInverse,
    pointer: &str,
    by: f64,
) -> Result<ApproxInverse, serde_json::Error> {
    let scale = |x: &mut serde_json::Value| *x = (x.as_f64().unwrap() * by).into();
    let mut v = serde_json::to_value(factor).unwrap();
    let edited = edit(&mut v, pointer, &mut |node| match node {
        serde_json::Value::Array(values) => (values.iter_mut())
            .filter(|x| x.as_f64().is_some_and(|f| f < 1.0))
            .for_each(scale),
        x => scale(x),
    });
    assert!(edited > 0, "the factor stores {pointer}");
    serde_json::from_value::<ApproxInverse>(v)
}

/// Each stored `|D_e⁻¹C|` row sums to at most one; past it the backward sum can overflow.
#[test]
fn an_elimination_whose_rows_sum_past_one_is_corrupt() {
    let rows: Vec<u32> = (0..300).map(|o| o % 60).collect();
    let cols: Vec<u32> = (0..300).map(|o| (o * 7 + o / 13) % 30).collect();
    let scales: Vec<f64> = (0..300).map(|o| 1.0 + (o % 5) as f64).collect();
    let gram = BipartiteFactorWidth2::scaled(
        FactorWidth1::new(60, &rows, None).unwrap(),
        FactorWidth1::new(30, &cols, None).unwrap(),
        &scales,
    )
    .unwrap();
    let factor = approx_chol::factorize(gram).unwrap();
    decode_scaled(&factor, "/cross/values", 1.0).expect("a stored elimination decodes");
    decode_scaled(&factor, "/cross/values", 2.0).expect_err("rows summing to two are corrupt");
}

/// Each column of a dense `|L|` and each step of an approximate one sums to at most one; past it the substitution can overflow.
#[rstest]
#[case::dense("/Dense", 24)]
#[case::approximate("/coefficients", 0)]
fn a_cholesky_whose_multipliers_pass_one_is_corrupt(
    #[case] stored: &str,
    #[case] exact_max_dim: usize,
) {
    let factor = grounded_grid(exact_max_dim);
    decode_scaled(&factor, stored, 1.0).expect("a stored Cholesky decodes");
    decode_scaled(&factor, stored, 2.0).expect_err("multipliers summing to two are corrupt");
}

/// A 4×4 grid grounded at a corner.
fn grounded_grid(exact_max_dim: usize) -> ApproxInverse {
    let (a, b): (Vec<u32>, Vec<u32>) = (0..16u32)
        .flat_map(|v| [(v, v + 1), (v, v + 4)])
        .filter(|&(v, w)| w < 16 && (w != v + 1 || w % 4 != 0))
        .unzip();
    let ground: Vec<f64> = (0..16).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let grid = Laplacian::new(16, &a, &b).unwrap();
    let grid = Sddm::new(grid, &ground).unwrap();
    let config = Config {
        exact_max_dim,
        ..Config::default()
    };
    factorize_with(grid, config, ScalingPolicy::default()).unwrap()
}

/// An approximate step's shares are weights over its degree; negated, they still bound `Σ|c| ≤ 1` but solve another matrix.
#[test]
fn an_approximate_negative_share_is_corrupt() {
    decode_scaled(&grounded_grid(0), "/coefficients", -1.0)
        .expect_err("a negative share is corrupt");
}

/// An approximate step's `D⁻¹` is positive, as every other stored pivot; a negated one solves indefinite.
#[rstest]
#[case::negative(-1.0)]
#[case::zero(0.0)]
fn an_approximate_inverse_degree_not_positive_is_corrupt(#[case] by: f64) {
    let (a, b): (Vec<u32>, Vec<u32>) = (0..29u32).map(|v| (v, v + 1)).unzip();
    let path = Laplacian::new(30, &a, &b).unwrap();
    let config = Config {
        exact_max_dim: 0,
        ..Config::default()
    };
    let factor: ApproxInverse = factorize_with(path, config, ScalingPolicy::default()).unwrap();
    decode_scaled(&factor, "/inverse_degree", by).expect_err("a D⁻¹ not positive is corrupt");
}

/// A factor holding an isolated vertex reads back from a self-describing format too.
#[test]
fn an_isolated_vertex_round_trips_through_json() {
    let factor = with_isolated_vertex();
    let json = serde_json::to_string(&factor).unwrap();
    let back: ApproxInverse = serde_json::from_str(&json).unwrap();
    let b = [1.0, 2.0, 3.0];
    assert_eq!(factor.solve(&b).unwrap(), back.solve(&b).unwrap());
}

/// The factor's first block stored as `tag`, in its JSON `[version, [n, dead, blocks, clamps]]`.
fn block_tagged<'a>(factor: &'a mut serde_json::Value, tag: &str) -> &'a mut serde_json::Value {
    (factor[1][2].as_array_mut().unwrap().iter_mut())
        .find(|block| block.get(tag).is_some())
        .unwrap()
}

/// Re-tags the factor's first `u32` coupled block as `u64`, over `vertices`.
fn widen_with(factor: &mut serde_json::Value, vertices: serde_json::Value) {
    let block = block_tagged(factor, "U32");
    let mut coupled = block["U32"].take();
    coupled["restriction"]["vertices"] = vertices;
    *block = serde_json::json!({ "U64": coupled });
}

/// A block's vertex `u64::MAX` is a decode error, never an overflow past the end of a run of consecutive vertices.
#[rstest]
#[case::isolated(&[u64::MAX])]
#[case::coupled(&[u64::MAX, 0])]
fn a_vertex_at_u64_max_is_corrupt(#[case] vertices: &[u64]) {
    let mut v = serde_json::to_value(with_isolated_vertex()).unwrap();
    match vertices {
        [vertex] => block_tagged(&mut v, "Isolated")["Isolated"]["vertex"] = (*vertex).into(),
        _ => widen_with(&mut v, vertices.into()),
    }
    serde_json::from_value::<ApproxInverse>(v).expect_err("no factor covers u64::MAX");
}

/// A lone `u64::MAX` for a two-vertex block is a decode error, never an overflow past it.
#[test]
fn a_lone_vertex_at_u64_max_for_two_is_corrupt() {
    let mut v = serde_json::to_value(with_isolated_vertex()).unwrap();
    widen_with(&mut v, serde_json::json!([u64::MAX]));
    serde_json::from_value::<ApproxInverse>(v).expect_err("one vertex for two is corrupt");
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
    serde_json::from_value::<ApproxInverse>(v).expect_err("empty scales are corrupt");
}

/// A factor of one unit path `3 → 2 → 1`, gauge `0`, its middle step `len` long.
fn path_block_bytes(len: u64) -> Vec<u8> {
    let mut bytes = one_block_factor(4);
    bytes.push(1); // Storage::U64
    bytes.extend([4, 0, 1, 2, 3, 0]); // vertices, Identity
    bytes.push(0); // no elimination
    bytes.extend([0, 4, 1, 0, 1]); // One, n = 4, pinned Some(0), Approximate
    bytes.push(3);
    for (vertex, len) in [(3, 1), (2, len), (1, 1)] {
        varint(vertex, &mut bytes);
        varint(len, &mut bytes);
        bytes.extend(1.0f64.to_le_bytes());
    }
    bytes.extend([3, 2, 1, 0, 3]);
    (0..3).for_each(|_| bytes.extend(1.0f64.to_le_bytes()));
    bytes.push(0); // no report
    bytes.push(0); // no clamp
    bytes
}

/// A forged u64 step length past the stored neighbors is a decode error, never an overflow; the same bytes at its true length decode.
#[test]
fn an_approximate_step_length_past_its_neighbors_is_corrupt() {
    postcard::from_bytes::<ApproxInverse>(&path_block_bytes(1)).expect("the true length decodes");
    postcard::from_bytes::<ApproxInverse>(&path_block_bytes(u64::MAX))
        .expect_err("a step past its neighbors is corrupt");
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
