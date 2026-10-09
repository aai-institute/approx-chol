use crate::Real;

/// `M`'s surplus `s = diag(M) − Σ|offdiag|`; floating means `M` is a Laplacian and singular.
#[derive(Clone, Debug)]
pub(crate) enum Ground<T> {
    Floating,
    Grounded(Surplus<T>),
}

impl<T: Real> Ground<T> {
    /// All-zero surplus floats; a round's complement grounds through here too.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        match Surplus::new(surplus) {
            Some(surplus) => Self::Grounded(surplus),
            None => Self::Floating,
        }
    }
}

/// Finite and nonnegative with a positive total: the only way to be grounded.
#[derive(Clone, Debug)]
pub(crate) struct Surplus<T>(Box<[T]>);

impl<T: Real> Surplus<T> {
    /// `None` for an all-zero surplus; panics on a negative or non-finite entry.
    fn new(values: Vec<T>) -> Option<Self> {
        todo!()
    }

    pub(crate) fn values(&self) -> &[T] {
        &self.0
    }
}
