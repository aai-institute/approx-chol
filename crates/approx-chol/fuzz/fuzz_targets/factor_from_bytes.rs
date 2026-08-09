#![no_main]

use approx_chol::{Factor, FACTOR_FORMAT_VERSION};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

/// Encoded once and prepended rather than fuzzed: a payload that does not open with these
/// exact bytes is rejected by one comparison, and a corpus spending its budget
/// rediscovering them never reaches the structural checks this target presses on.
fn version_prefix() -> &'static [u8] {
    static PREFIX: OnceLock<Vec<u8>> = OnceLock::new();
    PREFIX.get_or_init(|| postcard::to_stdvec(&FACTOR_FORMAT_VERSION).expect("encode the version"))
}

// Postcard is how `within` persists a `Factor` — inside a pickled preconditioner — so these
// bytes really do arrive from another process, and positionally.
fuzz_target!(|data: &[u8]| {
    let prefix = version_prefix();
    let mut payload = Vec::with_capacity(prefix.len() + data.len());
    payload.extend_from_slice(prefix);
    payload.extend_from_slice(data);
    let Ok(factor) = postcard::from_bytes::<Factor<f64>>(&payload) else {
        return;
    };
    // Deserializing is not the claim under test — a payload every structural check admits
    // and that then panics in `solve` is.
    let _ = factor.solve(&vec![1.0; factor.original_n()]);
    let mut values = vec![1.0; factor.n()];
    let _ = factor.solve_in_place(&mut values);
});
