#![cfg(feature = "serde")]

use approx_chol_next::input::Csr;
use approx_chol_next::{factorize, BlockFactor, Config, ScalingPolicy, Sgdd};
use rstest::rstest;
use serde::de::value::{Error, SeqDeserializer};
use serde::Deserialize;

/// The leading field this build writes.
const FORMAT_VERSION: u32 = 0x4143_0006;

/// Strictly dominant, so it reduces to one grounded block.
fn grounded_block() -> BlockFactor {
    let (row_ptrs, columns) = ([0u32, 2, 5, 7], [0u32, 1, 0, 1, 2, 1, 2]);
    let values = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
    let csr = Csr::new(3, &row_ptrs, &columns, &values).expect("valid csr");
    let sgdd: Sgdd = Sgdd::from_csr(csr, ScalingPolicy::default()).expect("reduces");
    let mut blocks = factorize(&sgdd, &Config::default()).into_blocks();
    blocks.pop().expect("one block")
}

#[test]
fn a_payload_declares_the_format_version_it_was_written_with() {
    let bytes = postcard::to_stdvec(&grounded_block()).expect("serialize");
    let (declared, _) = postcard::take_from_bytes::<u32>(&bytes).expect("a leading version");
    assert_eq!(declared, FORMAT_VERSION);
}

#[rstest]
#[case::from_the_old_crate(0x4143_0005)]
#[case::from_a_future_release(FORMAT_VERSION + 1)]
#[case::zero(0)]
fn a_payload_of_another_format_version_is_rejected_by_version(#[case] declared: u32) {
    let bytes = postcard::to_stdvec(&grounded_block()).expect("serialize");
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
