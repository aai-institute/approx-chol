use approx_chol::{ApplyError, ApproxInverse, Real, Threads};

/// One right-hand side through `apply`, into a new vector.
pub trait Solve<T> {
    fn solve_at(&self, b: &[T], threads: Threads) -> Result<Vec<T>, ApplyError>;

    fn solve(&self, b: &[T]) -> Result<Vec<T>, ApplyError> {
        self.solve_at(b, Threads::All)
    }
}

impl<T: Real> Solve<T> for ApproxInverse<T> {
    fn solve_at(&self, b: &[T], threads: Threads) -> Result<Vec<T>, ApplyError> {
        let mut x = b.to_vec();
        self.apply(&mut x, threads).map(|()| x)
    }
}
