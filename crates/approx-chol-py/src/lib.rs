use approx_chol::{factorize_with, Backend, Config, CsrRef, Factor, ScalingReport};
use num_traits::PrimInt;
use numpy::{BorrowError, Element, PyArray1, PyArrayDescr, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use std::error::Error;
use std::mem::size_of;

#[inline]
fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

/// The innermost cause, so the message names the defect and its row or vertex.
fn error_to_py(error: &dyn Error) -> PyErr {
    let mut cause = error;
    while let Some(source) = cause.source() {
        cause = source;
    }
    value_error(cause.to_string())
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

/// A CSR array's role, which fixes the dtype kinds it may hold without truncation.
#[derive(Clone, Copy)]
enum Column {
    Index,
    Value,
}

impl Column {
    fn accepts(self, kind: &str) -> bool {
        match self {
            Self::Index => kind == "i" || kind == "u",
            Self::Value => kind == "i" || kind == "u" || kind == "f",
        }
    }

    fn expected(self) -> &'static str {
        match self {
            Self::Index => "an integer dtype",
            Self::Value => "an integer or floating-point dtype",
        }
    }
}

/// `array_like` as a 1-D array of a kind `column` accepts, not yet cast or copied.
fn as_1d<'py>(
    np: &Bound<'py, PyModule>,
    array_like: &Bound<'py, PyAny>,
    name: &str,
    column: Column,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = np.call_method1("asarray", (array_like,))?;
    if arr.getattr("ndim")?.extract::<usize>()? != 1 {
        return Err(value_error(format!("{name} must be a 1-D array")));
    }
    let kind = arr.getattr("dtype")?.getattr("kind")?.extract::<String>()?;
    if !column.accepts(&kind) {
        return Err(value_error(format!(
            "{name} must have {}",
            column.expected()
        )));
    }
    Ok(arr)
}

/// `arr` as contiguous `E`, copied only when its dtype or layout differs.
fn as_contiguous<'py, E: Element>(
    np: &Bound<'py, PyModule>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<E>>> {
    np.call_method1("ascontiguousarray", (arr, E::get_dtype(arr.py())))?
        .cast_into()
        .map_err(Into::into)
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

#[pyclass(frozen, eq, from_py_object, name = "Backend")]
#[derive(Clone, PartialEq, Eq)]
enum PyBackend {
    Approximate {},
    ExactBelow { max_dim: usize },
}

impl From<PyBackend> for Backend {
    fn from(backend: PyBackend) -> Self {
        match backend {
            PyBackend::Approximate {} => Self::Approximate,
            PyBackend::ExactBelow { max_dim } => Self::ExactBelow { max_dim },
        }
    }
}

impl From<Backend> for PyBackend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::ExactBelow { max_dim } => Self::ExactBelow { max_dim },
            // `Backend` is `non_exhaustive`, so a newer core may route a way these bindings do not model yet.
            _ => Self::Approximate {},
        }
    }
}

#[pyclass(frozen, name = "Config")]
struct PyConfig(Config);

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=None, split=None, backend=None))]
    fn new(seed: Option<u64>, split: Option<u32>, backend: Option<PyBackend>) -> Self {
        let default = Config::default();
        Self(Config {
            seed: seed.unwrap_or(default.seed),
            split_merge: split.or(default.split_merge),
            backend: backend.map_or(default.backend, Backend::from),
            ..default
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed
    }

    #[getter]
    fn split(&self) -> Option<u32> {
        self.0.split_merge
    }

    #[getter]
    fn backend(&self) -> PyBackend {
        self.0.backend.into()
    }
}

#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: Factor<f64>,
}

/// Without this, a deficit the reduction clamped to make a block dominant is silent.
fn warn_on_scaling(py: Python<'_>, factor: &Factor<f64>) -> PyResult<()> {
    let reports: Vec<ScalingReport> = (factor.blocks().iter())
        .filter_map(|block| block.scaling_report())
        .collect();
    let clamped: Vec<f64> = (reports.iter())
        .map(|report| report.violation)
        .filter(|&violation| violation > 0.0)
        .collect();
    let unsettled = reports.iter().filter(|report| !report.settled).count();
    let mut causes = Vec::new();
    if !clamped.is_empty() {
        let worst = clamped.iter().fold(0.0f64, |worst, &v| worst.max(v));
        causes.push(format!(
            "{} block(s) fell short of diagonal dominance by up to {worst:.3e} \
             of a diagonal; the deficit was clamped",
            clamped.len()
        ));
    }
    if unsettled > 0 {
        causes.push(format!(
            "{unsettled} block(s) did not settle, so a singular one may have been grounded"
        ));
    }
    if causes.is_empty() {
        return Ok(());
    }
    let message = causes.join("; ");
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (message, py.get_type::<pyo3::exceptions::PyRuntimeWarning>()),
    )?;
    Ok(())
}

#[pymethods]
impl PyFactor {
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let n = self.inner.n();
        (n, n)
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        f64::get_dtype(py)
    }

    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let b_slice = b
            .as_slice()
            .map_err(|_| value_error("b must be contiguous"))?;
        let x = self.inner.solve(b_slice).map_err(|e| error_to_py(&e))?;
        Ok(PyArray1::from_vec(py, x))
    }

    fn solve_into<'py>(
        &self,
        b: PyReadonlyArray1<'py, f64>,
        out: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<()> {
        let b_slice = b
            .as_slice()
            .map_err(|_| value_error("b must be contiguous"))?;
        let n = self.inner.n();
        let out_ro = out.try_readonly().map_err(|e| borrow_error("out", e))?;
        let out_ro_slice = out_ro
            .as_slice()
            .map_err(|_| value_error("out must be contiguous"))?;
        if out_ro_slice.len() < n {
            return Err(value_error(format!(
                "out length {} is smaller than matrix dimension {}",
                out_ro_slice.len(),
                n
            )));
        }
        if slices_overlap(b_slice, out_ro_slice) {
            return Err(value_error("b and out must not overlap"));
        }
        drop(out_ro);
        if b_slice.len() != n {
            return Err(value_error(format!(
                "right-hand side has length {}, expected {n}",
                b_slice.len()
            )));
        }
        let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
        let x = &mut out_rw
            .as_slice_mut()
            .map_err(|_| value_error("out must be contiguous"))?[..n];
        x.copy_from_slice(b_slice);
        let mut scratch = vec![0.0; self.inner.scratch_len()];
        (self.inner.solve_in_place(x, &mut scratch)).map_err(|e| error_to_py(&e))
    }
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, n, config=None))]
fn factorize_raw<'py>(
    py: Python<'py>,
    row_ptrs: &Bound<'py, PyAny>,
    col_indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n: usize,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let np = py.import("numpy")?;
    CsrArrays {
        n,
        indptr: as_1d(&np, row_ptrs, "row_ptrs", Column::Index)?,
        indices: as_1d(&np, col_indices, "col_indices", Column::Index)?,
        data: as_contiguous(&np, &as_1d(&np, values, "values", Column::Value)?)?,
        np,
    }
    .factorize(config)
}

#[pyfunction]
#[pyo3(signature = (matrix, config=None))]
fn factorize(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let (n, columns) = matrix.getattr("shape")?.extract::<(usize, usize)>()?;
    if n != columns {
        return Err(value_error("matrix must be square"));
    }
    let np = py.import("numpy")?;
    let data = as_1d(&np, &matrix.getattr("data")?, "data", Column::Value)?;
    CsrArrays {
        n,
        indptr: as_1d(&np, &matrix.getattr("indptr")?, "indptr", Column::Index)?,
        indices: as_1d(&np, &matrix.getattr("indices")?, "indices", Column::Index)?,
        data: as_contiguous(&np, &data)?,
        np,
    }
    .factorize(config)
}

/// A CSR's checked arrays: values already `f64`, indices still in the caller's dtype.
struct CsrArrays<'py> {
    np: Bound<'py, PyModule>,
    n: usize,
    indptr: Bound<'py, PyAny>,
    indices: Bound<'py, PyAny>,
    data: Bound<'py, PyArray1<f64>>,
}

impl CsrArrays<'_> {
    fn factorize(&self, config: Option<&PyConfig>) -> PyResult<PyFactor> {
        // Also keeps `CsrRef::new` from overflowing `n + 1`.
        if u32::try_from(self.n).is_err() {
            return Err(value_error("matrix dimension exceeds u32::MAX"));
        }
        let dtypes = (
            self.indptr.getattr("dtype")?,
            self.indices.getattr("dtype")?,
        );
        let dtype = self.np.call_method1("result_type", dtypes)?;
        let kind = dtype.getattr("kind")?.extract::<String>()?;
        let width = dtype.getattr("itemsize")?.extract::<usize>()?;
        match (kind.as_str(), width) {
            ("i", 4) => self.factorize_as::<i32>(config),
            ("u", 4) => self.factorize_as::<u32>(config),
            ("u", 8) => self.factorize_as::<u64>(config),
            // Narrower integers widen; int64 beside uint64 promotes to float, and i64 holds any valid index.
            _ => self.factorize_as::<i64>(config),
        }
    }

    /// Borrows the index arrays whenever they already are contiguous `J`.
    fn factorize_as<J: Element + PrimInt + Sync>(
        &self,
        config: Option<&PyConfig>,
    ) -> PyResult<PyFactor> {
        let indptr = as_contiguous::<J>(&self.np, &self.indptr)?;
        let indices = as_contiguous::<J>(&self.np, &self.indices)?;
        // Borrowed arrays are the caller's, so another thread may hold one mutably.
        let indptr = indptr
            .try_readonly()
            .map_err(|e| borrow_error("indptr", e))?;
        let indices = indices
            .try_readonly()
            .map_err(|e| borrow_error("indices", e))?;
        let data = (self.data.try_readonly()).map_err(|e| borrow_error("data", e))?;
        let contiguous = |name: &str| value_error(format!("{name} must be contiguous"));
        let csr = CsrRef::new(
            indptr.as_slice().map_err(|_| contiguous("indptr"))?,
            indices.as_slice().map_err(|_| contiguous("indices"))?,
            data.as_slice().map_err(|_| contiguous("data"))?,
            self.n as u32,
        )
        .map_err(|e| error_to_py(&e))?;
        let config = config.map_or_else(Config::default, |config| config.0);
        let inner = factorize_with(csr, config).map_err(|e| error_to_py(&e))?;
        warn_on_scaling(self.np.py(), &inner)?;
        Ok(PyFactor { inner })
    }
}

/// Approximate Cholesky factorization for SDDM/Laplacian systems.
// GIL-free is sound: every pyclass is `frozen` with `Sync` state and the module holds none.
#[pymodule(gil_used = false)]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyFactor>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
