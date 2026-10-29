use approx_chol::{Config, SchurMode};

/// Every Cholesky dense, every Schur complement exact.
pub fn dense() -> Config {
    Config {
        complement: SchurMode::Exact,
        exact_max_dim: usize::MAX,
        ..Config::default()
    }
}
