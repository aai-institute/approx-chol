#![cfg(feature = "serde")]

use approx_chol::input::{BipartiteGram, FactorWidth1, Laplacian, SignedGraph, Weights};
use approx_chol::{factorize, BlockFactor, CliqueRule, Config, InnerParallelism, Sgdd};
use rstest::rstest;

fn inputs() -> Vec<Sgdd> {
    let (left, right, per_row) = (12_000u32, 400u32, 3u32);
    let rows: Vec<u32> = (0..left * per_row).map(|k| k / per_row).collect();
    let cols: Vec<u32> = (0..left * per_row)
        .map(|k| (k * 7 + k / per_row) % right)
        .collect();
    let slope: Vec<f64> = (0..rows.len())
        .map(|k| 1.0 + (k % 5) as f64 / 4.0)
        .collect();
    let gram = BipartiteGram::new(
        FactorWidth1::new(left as usize, &rows, Some(&slope)).unwrap(),
        FactorWidth1::new(right as usize, &cols, None).unwrap(),
        Weights::unit(),
    )
    .unwrap();
    let bipartite = Sgdd::from_bipartite_gram(gram, Default::default()).unwrap();

    let (a, b): (Vec<u32>, Vec<u32>) = (0..60u32)
        .filter(|i| i % 20 != 19)
        .map(|i| (i, i + 1))
        .unzip();
    let ground: Vec<f64> = (0..61).map(|i| if i < 20 { 0.0 } else { 0.5 }).collect();
    let laplacian = Laplacian::new(61, &a, &b, Weights::unit())
        .unwrap()
        .grounded(&ground)
        .unwrap();
    let laplacian = Sgdd::try_from(laplacian).unwrap();

    let (c, d): (Vec<u32>, Vec<u32>) = (0..58u32).map(|i| (i, i + 1 + i % 2)).unzip();
    let positive: Vec<bool> = (0..c.len()).map(|k| k % 3 == 0).collect();
    let signed = SignedGraph::new(60, &c, &d, &positive, Weights::unit()).unwrap();
    let signed = Sgdd::try_from(signed).unwrap();
    vec![bipartite, laplacian, signed]
}

fn configs() -> [Config; 2] {
    let exact = Config {
        complement: CliqueRule::Exact,
        dense_up_to: 0,
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
    let inputs = inputs();
    assert!(inputs
        .iter()
        .flat_map(Sgdd::blocks)
        .any(|block| block.floats()));
    let split = factorize(&inputs[0], &Config::default()).into_blocks();
    assert!(
        split.iter().any(|block| block.splittable_work() > 0),
        "a block keeps Cᵀ, which loading rebuilds"
    );
    for sgdd in inputs {
        for config in configs() {
            let blocks = factorize(&sgdd, &config).into_blocks();
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
    for sgdd in inputs() {
        let bytes =
            postcard::to_stdvec(&factorize(&sgdd, &Config::default()).into_blocks()).unwrap();
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

/// One block on vertices `[64, 0]` whose elimination has `rows` empty rows of `C`, cut off before its Cholesky.
fn truncated_elimination(rows: u64, n_cols: u64) -> Vec<u8> {
    let mut bytes = vec![1];
    varint(0x4143_0006, &mut bytes);
    bytes.extend([2, 64, 0, 0, 1]);
    varint(rows + 1, &mut bytes);
    bytes.extend(std::iter::repeat_n(0, rows as usize + 1));
    bytes.extend([0, 0]);
    varint(n_cols, &mut bytes);
    varint(rows, &mut bytes);
    (0..rows).for_each(|_| bytes.extend(1.0f64.to_le_bytes()));
    bytes.push(0);
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
