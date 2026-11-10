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

/// [`Fixture::Grounded`]'s body behind `declared`, written as postcard.
fn behind(declared: impl serde::Serialize) -> Vec<u8> {
    let bytes = postcard::to_stdvec(&Fixture::Grounded.factor()).expect("serialize");
    let (_, body) = postcard::take_from_bytes::<Header>(&bytes).expect("a header");
    let mut payload = postcard::to_stdvec(&declared).expect("serialize");
    payload.extend_from_slice(body);
    payload
}

#[rstest]
#[case::from_the_old_crate(
    Header { version: 0x4143_0005, scalar: Scalar::F64 },
    "format version 0x41430005, expected format version 0x4143000f"
)]
#[case::from_a_future_release(
    Header { version: 0x4143_0010, scalar: Scalar::F64 },
    "format version 0x41430010, expected format version 0x4143000f"
)]
#[case::zero(
    Header { version: 0, scalar: Scalar::F64 },
    "format version 0x00000000, expected format version 0x4143000f"
)]
#[case::of_f32(
    Header { version: 0x4143_000F, scalar: Scalar::F32 },
    "scalar f32, expected scalar f64"
)]
fn a_payload_of_another_header_is_rejected_by_name(#[case] declared: Header, #[case] named: &str) {
    let mismatch = declared.expect::<f64>().expect_err("another header");
    assert_eq!(mismatch.to_string(), named);
    assert_eq!(
        (mismatch.found, mismatch.expected),
        (declared, Header::of::<f64>())
    );
    assert_eq!(
        postcard::from_bytes::<ApproxInverse>(&behind(declared))
            .expect_err("another header must not decode"),
        postcard::Error::SerdeDeCustom
    );
    // Nothing valid follows the header, so only a gate that reads it first can name it.
    let header = serde_json::to_string(&declared).expect("serialize");
    for json in [
        format!("[{header}]"),
        format!(r#"[{header}, "not a factor"]"#),
    ] {
        let error = serde_json::from_str::<ApproxInverse>(&json)
            .expect_err("another header must not decode")
            .to_string();
        assert!(error.contains(named), "expected {named:?}, got: {error}");
    }
}

#[test]
fn an_f64_payload_is_rejected_as_f32_by_name() {
    let factor = Fixture::Grounded.factor();
    let bytes = postcard::to_stdvec(&factor).expect("serialize");
    assert_eq!(
        postcard::from_bytes::<ApproxInverse<f32>>(&bytes).expect_err("f64 is not f32"),
        postcard::Error::SerdeDeCustom
    );
    let json = serde_json::to_string(&factor).expect("serialize");
    let error = serde_json::from_str::<ApproxInverse<f32>>(&json)
        .expect_err("f64 is not f32")
        .to_string();
    assert!(error.contains("scalar f64, expected scalar f32"), "{error}");
}

/// A sequence is read in order, so a header wrong in both stops at its version; a map names both.
#[test]
fn a_header_wrong_in_both_is_rejected_by_version_first() {
    let declared = Header {
        version: 0,
        scalar: Scalar::F32,
    };
    let mismatch = declared.expect::<f64>().expect_err("another header");
    assert_eq!(
        mismatch.to_string(),
        "format version 0x00000000, expected format version 0x4143000f; \
         scalar f32, expected scalar f64"
    );
    let error = |json: &str| {
        serde_json::from_str::<ApproxInverse>(json)
            .expect_err("another header must not decode")
            .to_string()
    };
    let in_order = error(r#"[[0, "f32"]]"#);
    assert!(
        in_order.contains("format version 0x00000000, expected format version 0x4143000f")
            && !in_order.contains("scalar"),
        "{in_order}"
    );
    let both = error(r#"[{"version": 0, "scalar": "f32"}]"#);
    assert!(both.contains(&mismatch.to_string()), "{both}");
}

/// A load from 0.7.0, `(0x4143_0009, (n, dead, blocks))`, of an edge `0–1` grounded at `0` beside a lone vertex `2`.
const RELEASED_0_7: &str = "89808c8a04030002000200010000000102020003000000000000e03f000000000000e0bf0000000000000040000202000000000000e03f";
/// The same factor as `0x4143_000E` wrote it, `(version, B)` with no scalar.
const UNSCALARED_000E: &str = "8e808c8a0400020001000002000000000000e03f000000000000d03f";
/// A version alone.
const BARE_000E: &str = "8e808c8a04";

/// Postcard keeps no message; a gate reading the scalar or body first would fail `DeserializeUnexpectedEnd` on a bare version.
#[rstest]
#[case::released_0_7(RELEASED_0_7)]
#[case::unscalared_000e(UNSCALARED_000E)]
#[case::bare_version(BARE_000E)]
fn an_old_envelope_is_rejected_at_its_version(#[case] hex: &str) {
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect();
    assert_eq!(
        postcard::from_bytes::<ApproxInverse>(&bytes).expect_err("an old envelope"),
        postcard::Error::SerdeDeCustom
    );
}

/// 0.7.0's JSON of the factor in [`RELEASED_0_7`]: its version alone where a header stands.
const RELEASED_0_7_JSON: &str = r#"[1094909961,[3,[],[{"U32":{"restriction":{"vertices":[0,1],"congruence":"Identity"},"elimination":null,"cholesky":{"sheets":"One","grounded":true,"n":2,"pinned":2,"triangular":{"Dense":[0.5,-0.5,2.0]}},"report":null}},{"Isolated":{"vertex":2,"inverse_pivot":0.5}}]]]"#;

#[rstest]
#[case::released_0_7(
    RELEASED_0_7_JSON,
    "invalid type: integer `1094909961`, expected a Header"
)]
#[case::empty(r#"[[]]"#, "invalid length 0, expected a Header")]
#[case::version_alone(r#"[[1094909967]]"#, "invalid length 1, expected a Header")]
fn a_json_header_not_of_two_fields_is_rejected(#[case] json: &str, #[case] named: &str) {
    let error = serde_json::from_str::<ApproxInverse>(json)
        .expect_err("no header")
        .to_string();
    assert!(error.contains(named), "expected {named:?}, got: {error}");
}

#[test]
fn an_unknown_scalar_discriminant_is_corrupt() {
    let payload = behind((0x4143_000Fu32, 2u8));
    postcard::from_bytes::<ApproxInverse>(&payload).expect_err("no scalar 2");
}

#[rstest]
#[case::named(r#"[{"version": 1094909967, "scalar": "f16"}]"#)]
#[case::in_order(r#"[[1094909967, "f16"]]"#)]
fn an_unknown_scalar_name_is_rejected(#[case] json: &str) {
    let error = serde_json::from_str::<ApproxInverse>(json)
        .expect_err("no scalar f16")
        .to_string();
    assert!(error.contains("unknown variant `f16`"), "{error}");
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
