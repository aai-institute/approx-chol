use approx_chol::{Config, CsrRef, Error, SplitMerge};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

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
    #[pyo3(get)]
    merge: Option<u32>,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=None, merge=None))]
    fn new(seed: u64, split: Option<u32>, merge: Option<u32>) -> Self {
        Self { seed, split, merge }
    }
}

impl PyConfig {
    fn to_native(&self) -> Config {
        let split_merge = match (self.split, self.merge) {
            (Some(s), Some(m)) if s > 1 => Some(SplitMerge { split: s, merge: m }),
            (Some(s), None) if s > 1 => Some(SplitMerge { split: s, merge: s }),
            _ => None,
        };
        Config {
            seed: self.seed,
            split_merge,
        }
    }
}

// ---------------------------------------------------------------------------
// PyFactor
// ---------------------------------------------------------------------------

/// Approximate Cholesky factor (LDL^T decomposition).
#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: approx_chol::Factor<f64>,
}

#[pymethods]
impl PyFactor {
    /// Matrix dimension (may be larger than input if Gremban augmentation was applied).
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of elimination steps.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// Solve LDL^T x = b, returning a new numpy array.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let b_slice = b.as_slice().expect("contiguous array");
        let x = self.inner.solve(b_slice);
        PyArray1::from_vec(py, x)
    }

    /// Solve LDL^T x = b, writing the result into an existing array.
    fn solve_into<'py>(
        &self,
        b: PyReadonlyArray1<'py, f64>,
        out: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<()> {
        let b_slice = b.as_slice().expect("contiguous array");
        let mut out_rw = unsafe { out.as_array_mut() };
        let out_slice = out_rw
            .as_slice_mut()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("out must be contiguous"))?;
        self.inner.solve_into(b_slice, out_slice);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

fn approx_chol_err_to_py(e: Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
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
    let rp = row_ptrs.as_slice().expect("contiguous");
    let ci = col_indices.as_slice().expect("contiguous");
    let vals = values.as_slice().expect("contiguous");

    let csr = CsrRef::new(rp, ci, vals, n).map_err(approx_chol_err_to_py)?;
    let native_config = config.map_or_else(Config::default, |c| c.to_native());
    let factor = approx_chol::factorize_with(csr, native_config).map_err(approx_chol_err_to_py)?;
    Ok(PyFactor { inner: factor })
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must be square",
        ));
    }

    // Convert to the expected dtypes (uint32 for indices, float64 for data).
    let np = py.import("numpy")?;

    let rp_arr: PyReadonlyArray1<'_, u32> = np
        .call_method1("asarray", (&indptr,))?
        .call_method1("astype", (np.getattr("uint32")?,))?
        .extract()?;
    let ci_arr: PyReadonlyArray1<'_, u32> = np
        .call_method1("asarray", (&indices,))?
        .call_method1("astype", (np.getattr("uint32")?,))?
        .extract()?;
    let val_arr: PyReadonlyArray1<'_, f64> = np
        .call_method1("asarray", (&data,))?
        .call_method1("astype", (np.getattr("float64")?,))?
        .extract()?;

    let rp = rp_arr.as_slice().expect("contiguous");
    let ci = ci_arr.as_slice().expect("contiguous");
    let vals = val_arr.as_slice().expect("contiguous");

    let n = shape.0 as u32;
    let csr = CsrRef::new(rp, ci, vals, n).map_err(approx_chol_err_to_py)?;
    let native_config = config.map_or_else(Config::default, |c| c.to_native());
    let factor = approx_chol::factorize_with(csr, native_config).map_err(approx_chol_err_to_py)?;
    Ok(PyFactor { inner: factor })
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
