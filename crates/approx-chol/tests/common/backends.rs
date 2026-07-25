use approx_chol::Backend;

/// Behaviour that must hold however a block is factored. Test fixtures are small
/// enough that the default backend alone would leave the AC/AC2 sampler untested.
pub fn backends() -> [Backend; 2] {
    [Backend::Approximate, Backend::default()]
}
