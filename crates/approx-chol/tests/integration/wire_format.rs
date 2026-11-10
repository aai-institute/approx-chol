#![cfg(feature = "serde")]
//! Frozen payloads from earlier builds; `serde.rs` writes and reads with one build, so an encoding shift is invisible to it and visible here.

use crate::common::clamps::Clamps;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::Path;

use approx_chol::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
use approx_chol::{factorize_with, ApproxInverse, Config, FrustratedRoute, Header, Scalar};
use approx_chol::{ScalingOutcome, ScalingPolicy};
use rstest::rstest;
use serde::Deserialize;

use crate::common::pcg::relative_residual;
use crate::common::solve::Solve;
use crate::common::sparse::clique_tree;
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
    DenseFrustrated => dense_frustrated,
    SampledAc => sampled_ac,
    SampledAc2 => sampled_ac2,
    Clamped => clamped,
    Signed => signed,
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
            // A triangle with one positive edge: frustrated, so the approximate arm factors its double cover, the exact arm `M` itself, and the signed one `M` on one sheet, exactly, as its stars hold two neighbors.
            Self::Cover | Self::DenseFrustrated | Self::Signed => {
                Some(&[&[2.0, -1.0, 1.0], &[-1.0, 2.0, -1.0], &[1.0, -1.0, 2.0]])
            }
            // Two interleaved pairs no scaling makes dominant: each is clamped.
            Self::Clamped => Some(&[
                &[1.0, 0.0, -3.0, 0.0],
                &[0.0, 1.0, 0.0, -3.0],
                &[-3.0, 0.0, 1.0, 0.0],
                &[0.0, -3.0, 0.0, 1.0],
            ]),
            Self::SampledAc | Self::SampledAc2 => None,
        }
    }

    /// Whether the factor solves its matrix exactly: neither sampled nor clamped.
    fn exact(self) -> bool {
        !matches!(self, Self::SampledAc | Self::SampledAc2 | Self::Clamped)
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
        let csr = CsrRef::new(matrix.len(), &row_ptrs, &columns, &values).expect("valid csr");
        factorize_with(csr, self.config(), ScalingPolicy::default()).expect("factors")
    }

    fn config(self) -> Config {
        match self {
            Self::SampledAc => Config {
                exact_max_dim: 0,
                ..Config::default()
            },
            Self::SampledAc2 => Config {
                sampling: clique_tree(2),
                exact_max_dim: 0,
                ..Config::default()
            },
            Self::Cover | Self::GroundedApproximate => Config {
                exact_max_dim: 0,
                ..Config::default()
            },
            Self::Signed => Config {
                frustrated: FrustratedRoute::Signed,
                exact_max_dim: 0,
                ..Config::default()
            },
            _ => Config::default(),
        }
    }

    /// What the payload's `Debug` must show, so a fixture cannot quietly stop covering its kind.
    fn covers(self) -> &'static [&'static str] {
        match self {
            Self::Grounded => &["Dense([", "], None)"],
            Self::GroundedApproximate => &["Folded(", "len: 0", "] }, None)"],
            Self::Isolated => &["inverse_pivots: [0.5, 0.25]"],
            Self::Floating => &["n: 2, triangular: Dense(", "], Some("],
            Self::Cover => &["Cover("],
            Self::DenseFrustrated => &["n: 3, triangular: Dense(", "], None)"],
            Self::SampledAc | Self::SampledAc2 => &["elimination: Some", "Folded("],
            Self::Clamped => &["scaling: Refuted"],
            Self::Signed => &["n: 3, triangular: Signed("],
        }
    }

    /// The one place a fixture path is spelled, so the reader and the regenerator agree on the version.
    fn path(self) -> String {
        let version = Header::of::<f64>().version;
        format!("{FIXTURE_DIR}/{}_{version:#010x}.postcard", self.name())
    }
}

/// Small enough for AC2 to keep a few eliminations, large enough that its samples differ from AC's.
fn sampled_gram(config: Config) -> ApproxInverse {
    let (left, right, per_row) = (10u32, 6u32, 3u32);
    let (rows, cols): (Vec<u32>, Vec<u32>) = (0..left * per_row)
        .map(|k| (k / per_row, (k * 5 + k / per_row) % right))
        .unzip();
    let gram = BipartiteFactorWidth2::new(
        FactorWidth1::new(left as usize, &rows).expect("valid rows"),
        FactorWidth1::new(right as usize, &cols).expect("valid columns"),
    )
    .expect("valid gram");
    factorize_with(gram, config, ScalingPolicy::default()).expect("factors")
}

/// Zero-sum over every fixture's floating component, so each has an exact solution.
fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| [1.0, 2.0, -1.0, -2.0][i % 4]).collect()
}

/// `fixtures!` gives every fixture its case; this ties the committed payloads to those fixtures, so none is written but never read, an older version's included.
#[test]
fn every_fixture_has_a_case() {
    let file_name = |path: &Path| path.file_name().expect("a file").to_owned();
    let committed: BTreeSet<OsString> = std::fs::read_dir(FIXTURE_DIR)
        .expect("a fixtures directory")
        .map(|entry| entry.expect("a readable entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "postcard"))
        .map(|path| file_name(&path))
        .collect();
    let named: BTreeSet<OsString> = FIXTURES
        .iter()
        .map(|f| file_name(Path::new(&f.path())))
        .collect();
    assert_eq!(
        committed,
        named,
        "every committed payload is a fixture of format version {:#010x}",
        Header::of::<f64>().version
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
    for marker in fixture.covers() {
        assert!(
            built.contains(marker),
            "this build's {fixture:?} lost {marker:?}"
        );
    }
    let b = rhs(factor.n());
    let x = restored.solve_at(&b, threads::ONE).expect("rows fit");
    if let Some(matrix) = fixture.matrix().filter(|_| fixture.exact()) {
        let ax: Vec<f64> = (matrix.iter())
            .map(|row| row.iter().zip(&x).map(|(a, x)| a * x).sum())
            .collect();
        let residual = relative_residual(&ax, &b);
        assert!(residual < 1e-12, "relative residual {residual:e}");
    }
    // Each block's clamp, its outcome's variant order among it, is frozen with the format version.
    let restored_clamps = restored.clamps();
    let clamps: Vec<_> = (restored_clamps.iter())
        .map(|clamp| (clamp.vertices.as_slice(), clamp.scaling))
        .collect();
    let listed: Vec<_> = (fixture.clamps().iter())
        .map(|&vertices| (vertices, ScalingOutcome::Refuted))
        .collect();
    assert_eq!(clamps, listed, "{fixture:?}");
    assert_eq!(restored_clamps, factor.clamps(), "this build's clamps");

    // The residual alone is met by any valid factor, not only the one that wrote these bytes.
    let expected = factor.solve_at(&b, threads::ONE).expect("rows fit");
    assert!(
        x.iter()
            .zip(&expected)
            .all(|(got, want)| (got - want).abs() < 1e-12),
        "the committed payload solves differently from this build: {x:?} against {expected:?}"
    );
}

fn released(version: u32) -> Header {
    Header {
        version,
        ..Header::of::<f64>()
    }
}

#[rstest]
#[case::from_the_old_crate(released(0x4143_0005))]
#[case::from_a_future_release(released(Header::of::<f64>().version + 1))]
#[case::zero(released(0))]
#[case::of_f32(Header::of::<f32>())]
#[case::of_f32_and_zero(Header { version: 0, scalar: Scalar::F32 })]
fn a_payload_of_another_header_is_rejected_by_name(#[case] declared: Header) {
    let expected = Header::of::<f64>();
    let bytes = postcard::to_stdvec(&Fixture::Grounded.factor()).expect("serialize");
    let (_, body) = postcard::take_from_bytes::<(u32, Scalar)>(&bytes).expect("a flat header");
    let mut payload = postcard::to_stdvec(&(declared.version, declared.scalar)).expect("serialize");
    payload.extend_from_slice(body);
    assert_eq!(
        postcard::from_bytes::<ApproxInverse>(&payload)
            .expect_err("another header must not decode"),
        postcard::Error::SerdeDeCustom
    );

    let mismatch = declared.expect::<f64>().expect_err("another header");
    assert_eq!((mismatch.found, mismatch.expected), (declared, expected));
    // Nothing follows the header, so only a gate that reads it first can name it.
    let lone = serde_json::json!([declared]);
    let error = ApproxInverse::<f64>::deserialize(lone)
        .expect_err("another header must not decode")
        .to_string();
    let mut named = Vec::new();
    if declared.version != expected.version {
        named.push(format!(
            "format version {:#010x}, expected format version {:#010x}",
            declared.version, expected.version
        ));
    }
    if declared.scalar != expected.scalar {
        named.push("scalar f32, expected scalar f64".to_owned());
    }
    assert!(!named.is_empty());
    for named in named {
        assert!(error.contains(&named), "expected {named:?}, got: {error}");
        assert!(mismatch.to_string().contains(&named), "{mismatch}");
    }
}

/// A decoded clamp's violation is a deficit past rounding: finite and positive.
#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
#[case::negative_infinite(f64::NEG_INFINITY)]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
#[case::negative(-1.0)]
fn a_clamp_whose_violation_is_not_finite_and_positive_is_corrupt(#[case] violation: f64) {
    let factor = Fixture::Clamped.factor();
    let mut bytes = postcard::to_stdvec(&factor).expect("serialize");
    let stored = factor.clamps()[0].violation.to_le_bytes();
    let hits = bytes.windows(8).filter(|w| *w == stored).count();
    assert_eq!(
        hits,
        factor.clamps().len(),
        "only the clamps store {stored:?}"
    );
    let at = bytes.windows(8).position(|w| w == stored).expect("stored");
    bytes[at..at + 8].copy_from_slice(&violation.to_le_bytes());
    assert!(
        postcard::from_bytes::<ApproxInverse>(&bytes).is_err(),
        "{violation} decoded"
    );
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
