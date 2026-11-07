#![cfg(feature = "serde")]
//! Frozen payloads from earlier builds; `serde.rs` writes and reads with one build, so an encoding shift is invisible to it and visible here.

use std::collections::BTreeSet;

use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
use approx_chol::FACTOR_FORMAT_VERSION as FORMAT_VERSION;
use approx_chol::{factorize_with, ApproxInverse, Config, ScalingOutcome, ScalingPolicy};
use rstest::rstest;
use serde::de::value::{Error, SeqDeserializer};
use serde::Deserialize;

use crate::common::solve::Solve;
use crate::common::threads;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

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
    GroundedApproximate => grounded_approximate,
    Isolated => isolated,
    Floating => floating,
    Cover => cover,
    DenseCover => dense_cover,
    SampledAc2 => sampled_ac2,
    Clamped => clamped,
}

impl Fixture {
    /// The matrix a fixture factors, if not a sampled gram.
    fn matrix(self) -> Option<&'static [&'static [f64]]> {
        match self {
            // Strictly dominant, so it reduces to a grounded block.
            Self::Grounded => Some(&[&[2.0, -1.0, 0.0], &[-1.0, 3.0, -1.0], &[0.0, -1.0, 2.0]]),
            // A path grounded at every vertex: its ends share their pivots with the ground, and a tree factors exactly.
            Self::GroundedApproximate => Some(&[
                &[1.5, -1.0, 0.0, 0.0, 0.0, 0.0],
                &[-1.0, 2.5, -1.0, 0.0, 0.0, 0.0],
                &[0.0, -1.0, 2.5, -1.0, 0.0, 0.0],
                &[0.0, 0.0, -1.0, 2.5, -1.0, 0.0],
                &[0.0, 0.0, 0.0, -1.0, 2.5, -1.0],
                &[0.0, 0.0, 0.0, 0.0, -1.0, 1.5],
            ]),
            // Two vertices no entry couples, so each factor is its inverse pivot.
            Self::Isolated => Some(&[&[2.0, 0.0], &[0.0, 4.0]]),
            // Two interleaved floating components, so each block's vertices skip the other's.
            Self::Floating => Some(&[
                &[1.0, 0.0, -1.0, 0.0],
                &[0.0, 1.0, 0.0, -1.0],
                &[-1.0, 0.0, 1.0, 0.0],
                &[0.0, -1.0, 0.0, 1.0],
            ]),
            // A triangle with one positive edge: frustrated, so the approximate arm factors its double cover.
            Self::Cover | Self::DenseCover => {
                Some(&[&[2.0, -1.0, 1.0], &[-1.0, 2.0, -1.0], &[1.0, -1.0, 2.0]])
            }
            // Two interleaved pairs no scaling makes dominant: each is clamped.
            Self::Clamped => Some(&[
                &[1.0, 0.0, -3.0, 0.0],
                &[0.0, 1.0, 0.0, -3.0],
                &[-3.0, 0.0, 1.0, 0.0],
                &[0.0, -3.0, 0.0, 1.0],
            ]),
            Self::SampledAc2 => None,
        }
    }

    /// Whether the factor solves its matrix exactly: neither sampled nor clamped.
    fn exact(self) -> bool {
        !matches!(self, Self::SampledAc2 | Self::Clamped)
    }

    /// The clamps the payload lists, by vertices.
    fn clamps(self) -> &'static [&'static [usize]] {
        match self {
            Self::Clamped => &[&[0, 2], &[1, 3]],
            _ => &[],
        }
    }

    fn factor(self) -> ApproxInverse {
        let Some(matrix) = self.matrix() else {
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
        factorize_with(csr, self.config(), ScalingPolicy::default()).expect("factors")
    }

    fn config(self) -> Config {
        match self {
            Self::SampledAc2 => Config {
                split_merge: Some(2),
                exact_max_dim: 0,
                ..Config::default()
            },
            Self::Cover | Self::GroundedApproximate => Config {
                exact_max_dim: 0,
                ..Config::default()
            },
            _ => Config::default(),
        }
    }

    /// What the payload's `Debug` must show, so a fixture cannot quietly stop covering its kind.
    fn covers(self) -> &'static [&'static str] {
        match self {
            Self::Grounded => &["pinned: None"],
            Self::GroundedApproximate => &["pinned: None", "Approximate(", "len: 0"],
            Self::Isolated => &["Isolated("],
            Self::Floating => &["sheets: One, n: 2, pinned: Some("],
            Self::Cover => &["sheets: Two"],
            Self::DenseCover => &["sheets: Two", "Dense("],
            Self::SampledAc2 => &["elimination: Some", "Approximate("],
            Self::Clamped => &["scaling: Refuted"],
        }
    }

    /// What this build's factor must show: a legacy fixture's input no longer factors as its payload did.
    fn built(self) -> &'static [&'static str] {
        match self {
            // 0.7.0 factored a small frustrated block densely on its cover; it is now dense on one sheet.
            Self::DenseCover => &["sheets: One, n: 3, pinned: None", "Dense("],
            _ => self.covers(),
        }
    }

    /// The one place a fixture path is spelled, so the reader and the regenerator agree on the version.
    fn path(self) -> String {
        format!(
            "{FIXTURE_DIR}/{}_{FORMAT_VERSION:#010x}.postcard",
            self.name()
        )
    }
}

/// Small enough for AC2 to keep a few eliminations, large enough that its samples differ from AC's.
fn sampled_gram(config: Config) -> ApproxInverse {
    let (left, right, per_row) = (10u32, 6u32, 3u32);
    let (rows, cols): (Vec<u32>, Vec<u32>) = (0..left * per_row)
        .map(|k| (k / per_row, (k * 5 + k / per_row) % right))
        .unzip();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(left as usize, &rows, None).expect("valid rows"),
        FactorWidth1::new(right as usize, &cols, None).expect("valid columns"),
    )
    .expect("valid gram");
    factorize_with(gram, config, ScalingPolicy::default()).expect("factors")
}

/// Zero-sum over every fixture's floating component, so each has an exact solution.
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| [1.0, 2.0, -1.0, -2.0][i % 4]).collect()
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
    let committed: BTreeSet<String> = std::fs::read_dir(FIXTURE_DIR)
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
    let restored: ApproxInverse =
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
    let built = format!("{factor:?}");
    for marker in fixture.built() {
        assert!(
            built.contains(marker),
            "this build's {fixture:?} lost {marker:?}"
        );
    }
    let b = rhs(factor.n());
    let x = restored.solve_at(&b, threads::ONE).expect("rows fit");
    if let Some(matrix) = fixture.matrix().filter(|_| fixture.exact()) {
        let residual = relative_residual(matrix, &x, &b);
        assert!(residual < 1e-12, "relative residual {residual:e}");
    }
    // The clamp list's encoding, its outcome's variant order among it, is frozen with the format version.
    let clamps: Vec<_> = (restored.clamps().iter())
        .map(|clamp| (clamp.vertices.as_slice(), clamp.scaling))
        .collect();
    let listed: Vec<_> = (fixture.clamps().iter())
        .map(|&vertices| (vertices, ScalingOutcome::Refuted))
        .collect();
    assert_eq!(clamps, listed, "{fixture:?}");
    assert_eq!(restored.clamps(), factor.clamps(), "this build's clamps");

    // The residual alone is met by any valid factor, not only the one that wrote these bytes.
    let expected = factor.solve_at(&b, threads::ONE).expect("rows fit");
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
    let bytes = postcard::to_stdvec(&Fixture::Grounded.factor()).expect("serialize");
    let (_, body) = postcard::take_from_bytes::<u32>(&bytes).expect("a leading version");
    let mut payload = postcard::to_stdvec(&declared).expect("serialize the version");
    payload.extend_from_slice(body);
    assert_eq!(
        postcard::from_bytes::<ApproxInverse>(&payload)
            .expect_err("another version must not decode"),
        postcard::Error::SerdeDeCustom
    );

    // Nothing follows the version, so only a gate that reads it first can name it.
    let lone = SeqDeserializer::<_, Error>::new(std::iter::once(declared));
    let error = ApproxInverse::<f64>::deserialize(lone)
        .expect_err("another version must not decode")
        .to_string();
    for version in [declared, FORMAT_VERSION] {
        let named = format!("format version {version:#010x}");
        assert!(error.contains(&named), "expected {named:?}, got: {error}");
    }
}

/// After a version bump or a new fixture; a factorization change deletes and regenerates the affected fixture, with no version bump.
#[test]
#[ignore = "writes fixtures: cargo test -p approx-chol --features serde --test integration -- --ignored regenerate_missing_wire_format_fixtures"]
fn regenerate_missing_wire_format_fixtures() {
    for &fixture in FIXTURES {
        write_new(
            &fixture.path(),
            &postcard::to_stdvec(&fixture.factor()).expect("serialize"),
        );
    }
}

/// Rewriting a committed payload with today's encoder is how this test comes to agree with the drift it exists to catch.
fn write_new(path: &str, bytes: &[u8]) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path);
    match file {
        Ok(mut file) => {
            std::io::Write::write_all(&mut file, bytes).expect("write the fixture");
            println!("wrote {path}");
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => println!("kept {path}"),
        Err(error) => panic!("cannot create {path}: {error}"),
    }
}
