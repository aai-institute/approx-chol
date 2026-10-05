use approx_chol::{factorize_with, Config, CsrRef, Factor, Sddm, UnusablePivot};
use num_traits::{Float, PrimInt};

/// A CSR fixture that is SDDM by construction, converted and factored.
pub fn factor<T, I>(config: Config, csr: CsrRef<'_, T, I>) -> Result<Factor<T>, UnusablePivot>
where
    T: Float + Send + Sync + 'static,
    I: PrimInt,
{
    factorize_with(Sddm::try_from(csr).expect("fixture is SDDM"), config)
}
