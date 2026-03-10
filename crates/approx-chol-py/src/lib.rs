use approx_chol::{Config, CsrRef, Error};
use numpy::{BorrowError, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use std::mem::size_of;

#[inline]
fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[inline]
fn borrow_error(name: &str, err: BorrowError) -> PyErr {
    match err {
        BorrowError::AlreadyBorrowed => {
            pyo3::exceptions::PyBufferError::new_err(format!("{name} is already borrowed"))
        }
        BorrowError::NotWriteable => value_error(format!("{name} must be writeable")),
        _ => pyo3::exceptions::PyRuntimeError::new_err(format!("failed to borrow {name}: {err}")),
    }
}

fn validate_integer_index_array<'py>(
    np: &Bound<'py, PyModule>,
    array_like: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let arr = np.call_method1("asarray", (array_like,))?;
    let ndim = arr.getattr("ndim")?.extract::<usize>()?;
    if ndim != 1 {
        return Err(value_error(format!("{name} must be a 1-D array")));
    }
    let kind = arr.getattr("dtype")?.getattr("kind")?.extract::<String>()?;
    if kind != "i" && kind != "u" {
        return Err(value_error(format!("{name} must have an integer dtype")));
    }

    let size = arr.getattr("size")?.extract::<usize>()?;
    if size > 0 {
        let arr_i64 = arr.call_method1("astype", (np.getattr("int64")?,))?;
        let min_val = arr_i64.call_method0("min")?.extract::<i64>()?;
        if min_val < 0 {
            return Err(value_error(format!("{name} must be non-negative")));
        }
        let max_val = arr_i64.call_method0("max")?.extract::<i64>()?;
        if max_val > u32::MAX as i64 {
            return Err(value_error(format!("{name} exceeds u32::MAX")));
        }
    }

    let arr_u32 = arr.call_method1("astype", (np.getattr("uint32")?,))?;
    let arr_u32 = np.call_method1("ascontiguousarray", (arr_u32,))?;
    arr_u32.extract::<Bound<'py, PyArray1<u32>>>()
}

fn validate_values_array<'py>(
    np: &Bound<'py, PyModule>,
    array_like: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = np.call_method1("asarray", (array_like,))?;
    let ndim = arr.getattr("ndim")?.extract::<usize>()?;
    if ndim != 1 {
        return Err(value_error(format!("{name} must be a 1-D array")));
    }
    let kind = arr.getattr("dtype")?.getattr("kind")?.extract::<String>()?;
    if kind != "i" && kind != "u" && kind != "f" {
        return Err(value_error(format!(
            "{name} must have an integer or floating-point dtype"
        )));
    }
    let arr_f64 = arr.call_method1("astype", (np.getattr("float64")?,))?;
    let arr_f64 = np.call_method1("ascontiguousarray", (arr_f64,))?;
    arr_f64.extract::<Bound<'py, PyArray1<f64>>>()
}

#[inline]
fn slices_overlap<T>(lhs: &[T], rhs: &[T]) -> bool {
    if lhs.is_empty() || rhs.is_empty() {
        return false;
    }
    let lhs_start = lhs.as_ptr() as usize;
    let rhs_start = rhs.as_ptr() as usize;
    let lhs_end = lhs_start.saturating_add(lhs.len().saturating_mul(size_of::<T>()));
    let rhs_end = rhs_start.saturating_add(rhs.len().saturating_mul(size_of::<T>()));
    lhs_start < rhs_end && rhs_start < lhs_end
}

// ---------------------------------------------------------------------------
// PyConfig
// ---------------------------------------------------------------------------

/// Configuration for approximate Cholesky factorization.
#[pyclass(frozen, name = "Config")]
#[derive(Clone)]
struct PyConfig {
    #[pyo3(get)]
    seed: u64,
    #[pyo3(get)]
    split: Option<u32>,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=None))]
    fn new(seed: u64, split: Option<u32>) -> Self {
        Self { seed, split }
    }
}

impl PyConfig {
    fn to_native(&self) -> PyResult<Config> {
        let split_merge = match self.split {
            None | Some(1) => None,
            Some(0) => {
                return Err(value_error("config.split must be >= 1"));
            }
            Some(s) => Some(s),
        };
        Ok(Config {
            seed: self.seed,
            split_merge,
        })
    }
}

// ---------------------------------------------------------------------------
// PyFactor
// ---------------------------------------------------------------------------

/// Approximate Cholesky factor (LDL^T decomposition).
///
/// Implements the scipy `LinearOperator` duck-type interface (`shape`, `matvec`,
/// `rmatvec`, `dtype`) so it can be passed directly as `M=factor` to iterative
/// solvers like `scipy.sparse.linalg.cg`.
#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: approx_chol::Factor<f64>,
    /// Original matrix dimension (before possible Gremban augmentation).
    original_n: usize,
}

#[pymethods]
impl PyFactor {
    /// Internal factor dimension (may be larger than ``shape[0]`` if Gremban
    /// augmentation was applied).  Use this to size work buffers for
    /// :meth:`solve_into`.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of elimination steps.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// Preconditioner shape `(n, n)` reflecting the original matrix dimension.
    ///
    /// Part of the scipy `LinearOperator` duck-type interface.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.original_n, self.original_n)
    }

    /// Numpy dtype of output arrays (`numpy.float64`).
    ///
    /// Part of the scipy `LinearOperator` duck-type interface.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("float64")
    }

    /// Apply the preconditioner: solve LDL^T x = b, returning a vector of the
    /// original matrix dimension.
    ///
    /// Part of the scipy `LinearOperator` duck-type interface, enabling
    /// `M=factor` in iterative solvers like `scipy.sparse.linalg.cg`.
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    /// Alias for `matvec` (the LDL^T factor is symmetric).
    ///
    /// Part of the scipy `LinearOperator` duck-type interface.
    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    /// Solve LDL^T x = b, returning a new numpy array of the original
    /// matrix dimension.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let b_slice = b
            .as_slice()
            .map_err(|_| value_error("b must be contiguous"))?;
        let n = self.inner.n();
        if b_slice.len() > n {
            return Err(value_error(format!(
                "rhs length {} exceeds factor dimension {}",
                b_slice.len(),
                n
            )));
        }
        let x = self
            .inner
            .try_solve(b_slice)
            .map_err(|e| value_error(e.to_string()))?;
        Ok(PyArray1::from_vec(py, x))
    }

    /// Solve LDL^T x = b, writing the result into an existing array.
    ///
    /// The `out` array must have length >= `original_n` (the original matrix
    /// dimension, i.e. `factor.shape[0]`).
    fn solve_into<'py>(
        &self,
        b: PyReadonlyArray1<'py, f64>,
        out: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<()> {
        let b_slice = b
            .as_slice()
            .map_err(|_| value_error("b must be contiguous"))?;
        let n = self.inner.n();
        let original_n = self.original_n;
        if b_slice.len() > n {
            return Err(value_error(format!(
                "rhs length {} exceeds factor dimension {}",
                b_slice.len(),
                n
            )));
        }
        // Validate overlap and shape via shared borrows first. This prevents
        // taking a mutable NumPy view before proving it is safe to write.
        let out_ro = out.try_readonly().map_err(|e| borrow_error("out", e))?;
        let out_ro_slice = out_ro
            .as_slice()
            .map_err(|_| value_error("out must be contiguous"))?;
        if out_ro_slice.len() < original_n {
            return Err(value_error(format!(
                "out length {} is smaller than original matrix dimension {}",
                out_ro_slice.len(),
                original_n
            )));
        }
        if slices_overlap(b_slice, out_ro_slice) {
            return Err(value_error("b and out must not overlap"));
        }
        drop(out_ro);

        if original_n == n {
            // No augmentation: solve directly into out.
            let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
            let out_slice = out_rw
                .as_slice_mut()
                .map_err(|_| value_error("out must be contiguous"))?;
            self.inner
                .try_solve_into(b_slice, out_slice)
                .map_err(|e| value_error(e.to_string()))?;
        } else {
            // Augmented: solve into temp buffer, copy original_n elements.
            let mut work = vec![0.0; n];
            self.inner
                .try_solve_into(b_slice, &mut work)
                .map_err(|e| value_error(e.to_string()))?;
            let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
            let out_slice = out_rw
                .as_slice_mut()
                .map_err(|_| value_error("out must be contiguous"))?;
            out_slice[..original_n].copy_from_slice(&work[..original_n]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

fn approx_chol_err_to_py(e: Error) -> PyErr {
    value_error(e.to_string())
}

/// Factorize an SDDM matrix from raw CSR arrays.
///
/// Args:
///     row_ptrs: CSR row pointer array (uint32).
///     col_indices: CSR column index array (uint32).
///     values: CSR value array (float64).
///     n: Matrix dimension.
///     config: Optional factorization configuration.
#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, n, config=None))]
fn factorize_raw<'py>(
    row_ptrs: PyReadonlyArray1<'py, u32>,
    col_indices: PyReadonlyArray1<'py, u32>,
    values: PyReadonlyArray1<'py, f64>,
    n: u32,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let rp = row_ptrs
        .as_slice()
        .map_err(|_| value_error("row_ptrs must be contiguous"))?;
    let ci = col_indices
        .as_slice()
        .map_err(|_| value_error("col_indices must be contiguous"))?;
    let vals = values
        .as_slice()
        .map_err(|_| value_error("values must be contiguous"))?;

    let csr = CsrRef::new(rp, ci, vals, n).map_err(approx_chol_err_to_py)?;
    let original_n = n as usize;
    let factor = match config {
        Some(cfg) => {
            let native_config = cfg.to_native()?;
            approx_chol::factorize_with(csr, native_config)
        }
        None => approx_chol::factorize(csr),
    }
    .map_err(approx_chol_err_to_py)?;
    Ok(PyFactor {
        inner: factor,
        original_n,
    })
}

/// Factorize an SDDM matrix from a scipy.sparse CSR matrix.
///
/// Args:
///     matrix: A scipy.sparse.csr_matrix or csr_array.
///     config: Optional factorization configuration.
#[pyfunction]
#[pyo3(signature = (matrix, config=None))]
fn factorize(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let indptr = matrix.getattr("indptr")?.extract::<Bound<'_, PyAny>>()?;
    let indices = matrix.getattr("indices")?.extract::<Bound<'_, PyAny>>()?;
    let data = matrix.getattr("data")?.extract::<Bound<'_, PyAny>>()?;
    let shape = matrix.getattr("shape")?.extract::<(usize, usize)>()?;

    if shape.0 != shape.1 {
        return Err(value_error("matrix must be square"));
    }
    if shape.0 > u32::MAX as usize {
        return Err(value_error("matrix dimension exceeds u32::MAX"));
    }

    // Keep duck-typed input support but validate shape, dtype, and index ranges
    // before any narrowing conversions.
    let np = py.import("numpy")?;
    let rp_arr = validate_integer_index_array(&np, &indptr, "indptr")?;
    let ci_arr = validate_integer_index_array(&np, &indices, "indices")?;
    let val_arr = validate_values_array(&np, &data, "data")?;

    let rp_ro = rp_arr.readonly();
    let ci_ro = ci_arr.readonly();
    let vals_ro = val_arr.readonly();

    let rp = rp_ro
        .as_slice()
        .map_err(|_| value_error("indptr must be contiguous"))?;
    let ci = ci_ro
        .as_slice()
        .map_err(|_| value_error("indices must be contiguous"))?;
    let vals = vals_ro
        .as_slice()
        .map_err(|_| value_error("data must be contiguous"))?;

    let n = u32::try_from(shape.0).map_err(|_| value_error("matrix dimension exceeds u32::MAX"))?;
    let original_n = shape.0;
    let csr = CsrRef::new(rp, ci, vals, n).map_err(approx_chol_err_to_py)?;
    let factor = match config {
        Some(cfg) => {
            let native_config = cfg.to_native()?;
            approx_chol::factorize_with(csr, native_config)
        }
        None => approx_chol::factorize(csr),
    }
    .map_err(approx_chol_err_to_py)?;
    Ok(PyFactor {
        inner: factor,
        original_n,
    })
}

/// Approximate Cholesky factorization for SDDM/Laplacian systems.
#[pymodule]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyFactor>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
