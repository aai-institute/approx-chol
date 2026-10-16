use approx_chol::{Backend, Config, SchurMode};

/// Every Cholesky dense, every Schur complement exact.
pub fn dense() -> Config {
    Config {
        complement: SchurMode::Exact,
        backend: Backend::ExactBelow {
            max_dim: usize::MAX,
        },
        ..Config::default()
    }
}
