use approx_chol::Factor;

/// For input that factors as one block: it eliminates every input vertex exactly when a ground slot is left free.
pub fn is_grounded<T>(factor: &Factor<T>) -> bool
where
    T: num_traits::Float + Send + Sync + 'static,
{
    factor.n_steps() == factor.n()
}
