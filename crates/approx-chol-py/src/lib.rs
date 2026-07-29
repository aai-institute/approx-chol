use approx_chol::{Backend, Config, CsrRef, DenseFailure, Error, ExactFailure, Fallback};
use numpy::{BorrowError, Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
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

/// The element is the column's own associated type, so no call site can pair an index
/// column with a float element.
trait Column {
    type Element: numpy::Element;

    fn accepts(kind: &str) -> bool;

    fn expected() -> &'static str;

    fn check_range(_arr: &Bound<'_, PyAny>, _name: &str) -> PyResult<()> {
        Ok(())
    }
}

struct Index;

impl Column for Index {
    type Element = u32;

    fn accepts(kind: &str) -> bool {
        kind == "i" || kind == "u"
    }

    fn expected() -> &'static str {
        "an integer dtype"
    }

    fn check_range(arr: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
        if arr.getattr("size")?.extract::<usize>()? == 0 {
            return Ok(());
        }
        let wide = arr.call_method1("astype", (i64::get_dtype(arr.py()),))?;
        if wide.call_method0("min")?.extract::<i64>()? < 0 {
            return Err(value_error(format!("{name} must be non-negative")));
        }
        if wide.call_method0("max")?.extract::<i64>()? > u32::MAX as i64 {
            return Err(value_error(format!("{name} exceeds u32::MAX")));
        }
        Ok(())
    }
}

struct Value;

impl Column for Value {
    type Element = f64;

    fn accepts(kind: &str) -> bool {
        kind == "i" || kind == "u" || kind == "f"
    }

    fn expected() -> &'static str {
        "an integer or floating-point dtype"
    }
}

fn as_contiguous_1d<'py, C: Column>(
    np: &Bound<'py, PyModule>,
    array_like: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyArray1<C::Element>>> {
    let arr = np.call_method1("asarray", (array_like,))?;
    if arr.getattr("ndim")?.extract::<usize>()? != 1 {
        return Err(value_error(format!("{name} must be a 1-D array")));
    }
    let kind = arr.getattr("dtype")?.getattr("kind")?.extract::<String>()?;
    if !C::accepts(&kind) {
        return Err(value_error(format!("{name} must have {}", C::expected())));
    }
    C::check_range(&arr, name)?;
    let cast = arr.call_method1("astype", (C::Element::get_dtype(arr.py()),))?;
    np.call_method1("ascontiguousarray", (cast,))?.extract()
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

#[pyclass(frozen, eq, eq_int, name = "ExactFailure")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyExactFailure {
    FallBackToApproximate,
    Error,
}

#[pyclass(frozen, eq, name = "Backend")]
#[derive(Clone, PartialEq, Eq)]
enum PyBackend {
    Approximate {},
    ExactBelow {
        max_dim: usize,
        on_failure: PyExactFailure,
    },
}

impl From<PyBackend> for Backend {
    fn from(backend: PyBackend) -> Self {
        match backend {
            PyBackend::Approximate {} => Self::Approximate,
            PyBackend::ExactBelow {
                max_dim,
                on_failure,
            } => Self::ExactBelow {
                max_dim,
                on_failure: match on_failure {
                    PyExactFailure::FallBackToApproximate => ExactFailure::FallBackToApproximate,
                    PyExactFailure::Error => ExactFailure::Error,
                },
            },
        }
    }
}

impl Default for PyBackend {
    fn default() -> Self {
        match Backend::default() {
            Backend::ExactBelow {
                max_dim,
                on_failure,
            } => Self::ExactBelow {
                max_dim,
                // `ExactFailure` is `non_exhaustive`, so a newer core can default to a
                // policy these bindings do not model yet.
                on_failure: match on_failure {
                    ExactFailure::Error => PyExactFailure::Error,
                    _ => PyExactFailure::FallBackToApproximate,
                },
            },
            _ => Self::Approximate {},
        }
    }
}

#[pyclass(frozen, name = "Config")]
#[derive(Clone)]
struct PyConfig {
    #[pyo3(get)]
    seed: u64,
    #[pyo3(get)]
    split: Option<u32>,
    #[pyo3(get)]
    backend: PyBackend,
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=None, backend=None))]
    fn new(seed: u64, split: Option<u32>, backend: Option<PyBackend>) -> Self {
        Self {
            seed,
            split,
            backend: backend.unwrap_or_default(),
        }
    }
}

impl PyConfig {
    fn to_native(&self) -> Config {
        Config {
            seed: self.seed,
            split_merge: self.split,
            backend: self.backend.clone().into(),
        }
    }
}

#[pyclass(frozen, eq, eq_int, name = "DenseFailure")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyDenseFailure {
    NonPositivePivot,
    NonFinitePivot,
    Unknown,
}

impl From<DenseFailure> for PyDenseFailure {
    fn from(failure: DenseFailure) -> Self {
        // `DenseFailure` is `non_exhaustive`, so a newer core can report a cause
        // these bindings do not model yet.
        match failure {
            DenseFailure::NonPositivePivot => Self::NonPositivePivot,
            DenseFailure::NonFinitePivot => Self::NonFinitePivot,
            _ => Self::Unknown,
        }
    }
}

#[pyclass(frozen, eq, name = "Fallback")]
#[derive(Clone, PartialEq, Eq)]
enum PyFallback {
    InvalidPivot {
        vertex: usize,
        failure: PyDenseFailure,
    },
    WillNotFit {
        dim: usize,
    },
    // `Fallback` is `non_exhaustive`, so a newer core can report a reason these
    // bindings do not model yet.
    Other {
        reason: String,
    },
}

impl From<&Fallback> for PyFallback {
    fn from(fallback: &Fallback) -> Self {
        match *fallback {
            Fallback::InvalidPivot(pivot) => Self::InvalidPivot {
                vertex: pivot.vertex,
                failure: pivot.failure.into(),
            },
            Fallback::WillNotFit { dim } => Self::WillNotFit { dim },
            ref other => Self::Other {
                reason: other.to_string(),
            },
        }
    }
}

#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: approx_chol::Factor<f64>,
}

/// The default `ExactFailure` downgrades a block to approximate elimination, so
/// without this the accuracy loss is silent.
fn warn_on_fallback(py: Python<'_>, factor: &approx_chol::Factor<f64>) -> PyResult<()> {
    let fallbacks = factor.fallbacks();
    if fallbacks.is_empty() {
        return Ok(());
    }
    let detail = fallbacks
        .iter()
        .map(Fallback::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let message = format!(
        "{} block(s) routed to exact Cholesky fell back to approximate elimination: {detail}",
        fallbacks.len()
    );
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (message, py.get_type::<pyo3::exceptions::PyRuntimeWarning>()),
    )?;
    Ok(())
}

fn factorize_csr(
    py: Python<'_>,
    csr: CsrRef<'_, f64, u32>,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let config = config.map(PyConfig::to_native).unwrap_or_default();
    let inner = approx_chol::factorize_with(csr, config).map_err(approx_chol_err_to_py)?;
    warn_on_fallback(py, &inner)?;
    Ok(PyFactor { inner })
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
    fn fallbacks(&self) -> Vec<PyFallback> {
        self.inner
            .fallbacks()
            .iter()
            .map(PyFallback::from)
            .collect()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let n = self.inner.original_n();
        (n, n)
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("float64")
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
        let x = self
            .inner
            .solve(b_slice)
            .map_err(|e| value_error(e.to_string()))?;
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
        let original_n = self.inner.original_n();
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

        let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| value_error("out must be contiguous"))?;
        // `out` is only guaranteed to hold `original_n`, so a ground vertex — the one
        // variable past it — has to land in scratch rather than past the caller's end.
        if original_n == n {
            self.inner
                .solve_into(b_slice, out_slice)
                .map_err(|e| value_error(e.to_string()))?;
        } else {
            let mut work = vec![0.0; n];
            self.inner
                .solve_into(b_slice, &mut work)
                .map_err(|e| value_error(e.to_string()))?;
            out_slice[..original_n].copy_from_slice(&work[..original_n]);
        }
        Ok(())
    }
}

fn approx_chol_err_to_py(e: Error) -> PyErr {
    value_error(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, n, config=None))]
fn factorize_raw<'py>(
    py: Python<'py>,
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
    factorize_csr(py, csr, config)
}

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

    let np = py.import("numpy")?;
    let rp_arr = as_contiguous_1d::<Index>(&np, &indptr, "indptr")?;
    let ci_arr = as_contiguous_1d::<Index>(&np, &indices, "indices")?;
    let val_arr = as_contiguous_1d::<Value>(&np, &data, "data")?;

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
    let csr = CsrRef::new(rp, ci, vals, n).map_err(approx_chol_err_to_py)?;
    factorize_csr(py, csr, config)
}

#[pymodule]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyDenseFailure>()?;
    m.add_class::<PyExactFailure>()?;
    m.add_class::<PyFactor>()?;
    m.add_class::<PyFallback>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
