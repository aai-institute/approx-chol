use approx_chol::{MatrixMut, Real};

/// Columns stored apart, reached only through `read_column` and `write_column`, as a block of the caller's own type is.
#[derive(Clone, Debug, PartialEq)]
pub struct Apart<T>(pub Vec<Vec<T>>);

impl<T: Real> MatrixMut<T> for Apart<T> {
    fn rows(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }

    fn columns(&self) -> usize {
        self.0.len()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(&self.0[column]);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.0[column].copy_from_slice(from);
    }
}
