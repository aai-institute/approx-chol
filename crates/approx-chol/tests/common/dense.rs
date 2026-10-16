use approx_chol::{CliqueRule, Config};

/// Every Cholesky dense, every Schur complement exact.
pub fn dense() -> Config {
    Config {
        complement: CliqueRule::Exact,
        dense_up_to: usize::MAX,
        ..Config::default()
    }
}
