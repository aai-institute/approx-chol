#![cfg(feature = "serde")]
//! Frozen payloads from earlier builds; `serde.rs` writes and reads with one build, so an encoding shift is invisible to it and visible here.

use std::collections::BTreeSet;

use approx_chol::input::{BipartiteGram, CsrRef, FactorWidth1, Weights};
use approx_chol::{factorize_with, Backend, BlockFactor, Config, Factor, InnerParallelism};
use rstest::rstest;
use serde::de::value::{Error, SeqDeserializer};
use serde::Deserialize;

/// The leading field this build writes; a bump points the reader at new fixture files.
const FORMAT_VERSION: u32 = 0x4143_0006;

/// One list declares each fixture, its file name and its rstest case, so none is kept in step by hand.
macro_rules! fixtures {
    ($($variant:ident => $name:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug)]
        enum Fixture {
            $($variant),+
        }

        const FIXTURES: &[Fixture] = &[$(Fixture::$variant),+];

        impl Fixture {
            fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($name)),+
                }
            }
        }

        #[rstest]
        $(#[case::$name(Fixture::$variant)])+
        fn a_committed_payload_decodes_and_still_solves(#[case] fixture: Fixture) {
            assert_committed_payload_decodes_and_solves(fixture);
        }
    };
}

fixtures! {
    Grounded => grounded,
    Floating => floating,
    Cover => cover,
    SampledAc2 => sampled_ac2,
}

impl Fixture {
    /// The matrix of a fixture factored exactly; a sampled factor only preconditions, so it has none to meet.
    fn exact_matrix(self) -> Option<&'static [&'static [f64]]> {
        match self {
            // Strictly dominant, so it reduces to a grounded block.
            Self::Grounded => Some(&[&[2.0, -1.0, 0.0], &[-1.0, 3.0, -1.0], &[0.0, -1.0, 2.0]]),
            // Two interleaved floating components, so each block's vertices skip the other's.
            Self::Floating => Some(&[
                &[1.0, 0.0, -1.0, 0.0],
                &[0.0, 1.0, 0.0, -1.0],
                &[-1.0, 0.0, 1.0, 0.0],
                &[0.0, -1.0, 0.0, 1.0],
            ]),
            // A triangle with one positive edge: frustrated, so it factors on the double cover.
            Self::Cover => Some(&[&[2.0, -1.0, 1.0], &[-1.0, 2.0, -1.0], &[1.0, -1.0, 2.0]]),
            Self::SampledAc2 => None,
        }
    }

    fn factor(self) -> Factor {
        let Some(matrix) = self.exact_matrix() else {
            return sampled_gram(self.config());
        };
        let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
        for row in matrix {
            for (j, &value) in row.iter().enumerate().filter(|(_, &v)| v != 0.0) {
                columns.push(j as u32);
                values.push(value);
            }
            row_ptrs.push(columns.len() as u32);
        }
        let n = matrix.len() as u32;
        let csr = CsrRef::new(&row_ptrs, &columns, &values, n).expect("valid csr");
        factorize_with(csr, self.config()).expect("factors")
    }

    fn config(self) -> Config {
        match self {
            Self::SampledAc2 => Config {
                split_merge: Some(2),
                backend: Backend::Approximate,
                ..Config::default()
            },
            _ => Config::default(),
        }
    }

    /// What the payload's `Debug` must show, so a fixture cannot quietly stop covering its kind.
    fn covers(self) -> &'static [&'static str] {
        match self {
            Self::Grounded => &["grounded: true"],
            Self::Floating => &["sheets: One, grounded: false"],
            Self::Cover => &["sheets: Two"],
            Self::SampledAc2 => &["elimination: Some", "Approximate("],
        }
    }

    /// The one place a fixture path is spelled, so the reader and the regenerator agree on the version.
    fn path(self) -> String {
        format!(
            "{}/tests/fixtures/{}_{FORMAT_VERSION:#010x}.postcard",
            env!("CARGO_MANIFEST_DIR"),
            self.name()
        )
    }

    fn blocks(self) -> Vec<BlockFactor> {
        self.factor().into_blocks()
    }
}

/// Small enough for AC2 to keep a few eliminations, large enough that its samples differ from AC's.
fn sampled_gram(config: Config) -> Factor {
    let (left, right, per_row) = (10u32, 6u32, 3u32);
    let (rows, cols): (Vec<u32>, Vec<u32>) = (0..left * per_row)
        .map(|k| (k / per_row, (k * 5 + k / per_row) % right))
        .unzip();
    let gram = BipartiteGram::new(
        FactorWidth1::new(left as usize, &rows, None).expect("valid rows"),
        FactorWidth1::new(right as usize, &cols, None).expect("valid columns"),
        Weights::unit(),
    )
    .expect("valid gram");
    factorize_with(gram, config).expect("factors")
}

/// Zero-sum over every fixture's floating component, so each has an exact solution.
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| [1.0, 2.0, -1.0, -2.0][i % 4]).collect()
}

fn solve(blocks: &[BlockFactor], b: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; b.len()];
    for block in blocks {
        let vertices: Vec<usize> = block.vertices().collect();
        let mut local: Vec<f64> = vertices.iter().map(|&v| b[v]).collect();
        let mut scratch = vec![0.0; block.scratch_len()];
        (block.solve_local(&mut local, &mut scratch, InnerParallelism::Forbid))
            .expect("buffers fit");
        vertices.iter().zip(local).for_each(|(&v, xv)| x[v] = xv);
    }
    x
}

fn relative_residual(matrix: &[&[f64]], x: &[f64], b: &[f64]) -> f64 {
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|e| e * e).sum::<f64>().sqrt();
    let mut error = matrix.iter().zip(b).map(|(row, bi)| {
        let ax: f64 = row.iter().zip(x).map(|(a, xj)| a * xj).sum();
        ax - bi
    });
    norm(&mut error) / norm(&mut b.iter().copied())
}

/// `fixtures!` gives every fixture its case; this ties the committed payloads to those fixtures, so none is written but never read.
#[test]
fn every_fixture_has_a_case() {
    let suffix = format!("_{FORMAT_VERSION:#010x}.postcard");
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let committed: BTreeSet<String> = std::fs::read_dir(directory)
        .expect("a fixtures directory")
        .map(|entry| entry.expect("a readable entry").file_name())
        .filter_map(|name| Some(name.to_str()?.strip_suffix(&suffix)?.to_owned()))
        .collect();
    let named: BTreeSet<String> = FIXTURES.iter().map(|f| f.name().to_owned()).collect();
    assert_eq!(
        committed, named,
        "every committed payload of this version is a fixture"
    );
}

fn assert_committed_payload_decodes_and_solves(fixture: Fixture) {
    let committed = std::fs::read(fixture.path())
        .expect("no committed payload for this version; regenerate if the version moved");
    let restored: Vec<BlockFactor> =
        postcard::from_bytes(&committed).expect("committed payload must decode");
    assert_eq!(
        postcard::to_stdvec(&restored).expect("serialize"),
        committed
    );
    let shown = format!("{restored:?}");
    for marker in fixture.covers() {
        assert!(shown.contains(marker), "{fixture:?} lost {marker:?}");
    }

    let factor = fixture.factor();
    let b = rhs(factor.n());
    let x = solve(&restored, &b);
    if let Some(matrix) = fixture.exact_matrix() {
        let residual = relative_residual(matrix, &x, &b);
        assert!(residual < 1e-12, "relative residual {residual:e}");
    }

    // The residual alone is met by any valid factor, not only the one that wrote these bytes.
    let expected = solve(&factor.into_blocks(), &b);
    assert!(
        x.iter()
            .zip(&expected)
            .all(|(got, want)| (got - want).abs() < 1e-12),
        "the committed payload solves differently from this build: {x:?} against {expected:?}"
    );
}

#[rstest]
#[case::from_the_old_crate(0x4143_0005)]
#[case::from_a_future_release(FORMAT_VERSION + 1)]
#[case::zero(0)]
fn a_payload_of_another_format_version_is_rejected_by_version(#[case] declared: u32) {
    let block = Fixture::Grounded.blocks().pop().expect("one block");
    let bytes = postcard::to_stdvec(&block).expect("serialize");
    let (_, body) = postcard::take_from_bytes::<u32>(&bytes).expect("a leading version");
    let mut payload = postcard::to_stdvec(&declared).expect("serialize the version");
    payload.extend_from_slice(body);
    assert_eq!(
        postcard::from_bytes::<BlockFactor>(&payload).expect_err("another version must not decode"),
        postcard::Error::SerdeDeCustom
    );

    // Nothing follows the version, so only a gate that reads it first can name it.
    let lone = SeqDeserializer::<_, Error>::new(std::iter::once(declared));
    let error = BlockFactor::<f64>::deserialize(lone)
        .expect_err("another version must not decode")
        .to_string();
    for version in [declared, FORMAT_VERSION] {
        let named = format!("format version {version:#010x}");
        assert!(error.contains(&named), "expected {named:?}, got: {error}");
    }
}

/// After a version bump or a new fixture; a factorization change deletes and regenerates the affected fixture, with no version bump.
#[test]
#[ignore = "writes fixtures: cargo test -p approx-chol --features serde --test wire_format -- --ignored"]
fn regenerate_missing_wire_format_fixtures() {
    for &fixture in FIXTURES {
        let path = fixture.path();
        let bytes = postcard::to_stdvec(&fixture.blocks()).expect("serialize the fixture");
        // Rewriting a committed payload with today's encoder is how this test comes to agree with the drift it exists to catch.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path);
        match file {
            Ok(mut file) => {
                std::io::Write::write_all(&mut file, &bytes).expect("write the fixture");
                println!("wrote {path}");
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                println!("kept {path}");
            }
            Err(error) => panic!("cannot create {path}: {error}"),
        }
    }
}
