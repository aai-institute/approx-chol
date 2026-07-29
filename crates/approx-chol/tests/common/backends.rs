use approx_chol::Backend;

/// Every fixture here is small enough that the default backend routes it to exact
/// Cholesky, so the default alone leaves the AC/AC2 sampler untested.
pub fn backends() -> [Backend; 2] {
    [Backend::Approximate, Backend::default()]
}
