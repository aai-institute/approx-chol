use approx_chol::{
    Backend, Config, CsrRef, Factor, ScalingPolicy, ScalingReport, Sgdd, SolveError,
};
use num_traits::PrimInt;
use numpy::{
    BorrowError, Element, PyArray1, PyArrayDescr, PyArrayMethods, PyReadonlyArray1,
    PyUntypedArrayMethods,
};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::error::Error;

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

/// A matrix's stored entries, which size every call on it and its factor.
#[derive(Clone, Copy)]
struct StoredEntries(usize);

impl StoredEntries {
    /// Measured: up to 2^15 entries, a handoff to a busy thread costs more than parallel calls gain.
    fn detach<T: Ungil>(self, py: Python<'_>, work: impl Ungil + FnOnce() -> T) -> T {
        if self.0 > 1 << 15 {
            py.detach(work)
        } else {
            work()
        }
    }
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

/// Any 1-D array-like, or scipy's `(n, 1)` column, as contiguous `f64`; a CG's own `f64` vector skips every numpy call.
fn right_hand_side<'py>(
    py: Python<'py>,
    b: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let b = match b.cast::<PyArray1<f64>>() {
        Ok(array) if array.is_contiguous() => array.clone(),
        _ => {
            let np = py.import("numpy")?;
            let b = np.call_method1("asarray", (b,))?;
            // LinearOperator's default matmat before scipy 1.18 passes columns and reshapes the result itself.
            let column = b.getattr("ndim")?.extract::<usize>()? == 2
                && b.getattr("shape")?.get_item(1)?.extract::<usize>()? == 1;
            let b = if column { b.call_method0("ravel")? } else { b };
            as_contiguous(&np, &as_1d(&np, &b, "b", Column::Value)?)?
        }
    };
    b.try_readonly().map_err(|e| borrow_error("b", e))
}

#[inline]
/// A contiguous copy, its borrow released on return so the caller's array is free again.
fn owned<T: Element + Copy>(array: &Bound<'_, PyArray1<T>>, name: &str) -> PyResult<Vec<T>> {
    let array = array.try_readonly().map_err(|e| borrow_error(name, e))?;
    let slice = array.as_slice();
    slice
        .map(<[T]>::to_vec)
        .map_err(|_| value_error(format!("{name} must be contiguous")))
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

#[pyclass(frozen, eq, name = "ScalingPolicy")]
#[derive(PartialEq)]
struct PyScalingPolicy(ScalingPolicy);

#[pymethods]
impl PyScalingPolicy {
    #[new]
    #[pyo3(signature = (tolerance=None, max_iterations=None))]
    fn new(tolerance: Option<f64>, max_iterations: Option<usize>) -> PyResult<Self> {
        let default = ScalingPolicy::default();
        ScalingPolicy::new(
            tolerance.unwrap_or(default.tolerance()),
            max_iterations.unwrap_or(default.max_iterations()),
        )
        .map(Self)
        .ok_or_else(|| value_error("tolerance must be in [0, 1)"))
    }

    #[getter]
    fn tolerance(&self) -> f64 {
        self.0.tolerance()
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.0.max_iterations()
    }
}

#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: Factor<f64>,
    entries: StoredEntries,
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
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    fn rmatvec<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.solve(py, x)
    }

    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let b = right_hand_side(py, b)?;
        let b_slice = b.as_slice().expect("right_hand_side is contiguous");
        let (mut x, mut scratch) = (b_slice.to_vec(), vec![0.0; self.inner.scratch_len()]);
        let solved = (self.entries).detach(py, || self.inner.solve_in_place(&mut x, &mut scratch));
        solved.map_err(|e| error_to_py(&e))?;
        Ok(PyArray1::from_vec(py, x))
    }

    fn workspace(&self) -> PyWorkspace {
        PyWorkspace {
            x: Vec::new(),
            scratch: vec![0.0; self.inner.scratch_len()],
        }
    }

    #[pyo3(signature = (b, out, workspace=None))]
    fn solve_into<'py>(
        &self,
        py: Python<'py>,
        b: &Bound<'py, PyAny>,
        out: &Bound<'py, PyArray1<f64>>,
        mut workspace: Option<PyRefMut<'py, PyWorkspace>>,
    ) -> PyResult<()> {
        let n = self.inner.n();
        let mut owned = None;
        let PyWorkspace { x, scratch } = match workspace.as_deref_mut() {
            Some(workspace) => workspace,
            None => owned.insert(self.workspace()),
        };
        {
            let b = right_hand_side(py, b)?;
            x.clear();
            x.extend_from_slice(b.as_slice().expect("right_hand_side is contiguous"));
        }
        // `b`'s borrow is released, so `out` may be `b` or overlap it.
        let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
        let out = out_rw
            .as_slice_mut()
            .map_err(|_| value_error("out must be contiguous"))?;
        if out.len() < n {
            return Err(value_error(format!(
                "out length {} is smaller than matrix dimension {n}",
                out.len()
            )));
        }
        let out = &mut out[..n];
        // Our own buffer: without the GIL, or on a free-threaded build, another thread may touch `out`.
        let solved = (self.entries).detach(py, || self.inner.solve_in_place(x, scratch));
        solved
            .map(|()| out.copy_from_slice(x))
            .map_err(|e| match e {
                SolveError::ScratchTooShort { .. } => {
                    value_error("workspace is from a smaller factor; use this factor's workspace()")
                }
                e => error_to_py(&e),
            })
    }
}

/// Caller-owned solve buffers, so repeated `solve_into` calls allocate nothing.
#[pyclass(name = "Workspace")]
struct PyWorkspace {
    /// Grown by the first solve; `workspace()` allocates only `scratch`.
    x: Vec<f64>,
    scratch: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, n, config=None, scaling=None))]
fn factorize_raw<'py>(
    py: Python<'py>,
    row_ptrs: &Bound<'py, PyAny>,
    col_indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n: usize,
    config: Option<&PyConfig>,
    scaling: Option<&PyScalingPolicy>,
) -> PyResult<PyFactor> {
    let np = py.import("numpy")?;
    CsrArrays {
        n,
        indptr: as_1d(&np, row_ptrs, "row_ptrs", Column::Index)?,
        indices: as_1d(&np, col_indices, "col_indices", Column::Index)?,
        data: as_contiguous(&np, &as_1d(&np, values, "values", Column::Value)?)?,
        np,
    }
    .factorize(config, scaling)
}

#[pyfunction]
#[pyo3(signature = (matrix, config=None, scaling=None))]
fn factorize(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    config: Option<&PyConfig>,
    scaling: Option<&PyScalingPolicy>,
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
    .factorize(config, scaling)
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
    fn factorize(
        &self,
        config: Option<&PyConfig>,
        scaling: Option<&PyScalingPolicy>,
    ) -> PyResult<PyFactor> {
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
            ("i", 4) => self.factorize_as::<i32>(config, scaling),
            ("u", 4) => self.factorize_as::<u32>(config, scaling),
            ("u", 8) => self.factorize_as::<u64>(config, scaling),
            // Narrower integers widen; int64 beside uint64 promotes to float, and i64 holds any valid index.
            _ => self.factorize_as::<i64>(config, scaling),
        }
    }

    /// Copies the arrays, so nothing after reads a caller's array.
    fn factorize_as<J: Element + PrimInt + Sync + Send>(
        &self,
        config: Option<&PyConfig>,
        scaling: Option<&PyScalingPolicy>,
    ) -> PyResult<PyFactor> {
        let indptr = as_contiguous::<J>(&self.np, &self.indptr)?;
        let indices = as_contiguous::<J>(&self.np, &self.indices)?;
        let indptr = owned(&indptr, "indptr")?;
        let indices = owned(&indices, "indices")?;
        let data = owned(&self.data, "data")?;
        let config = config.map_or_else(Config::default, |config| config.0);
        let scaling = scaling.map_or_else(ScalingPolicy::default, |scaling| scaling.0);
        let (py, n, entries) = (self.np.py(), self.n as u32, StoredEntries(data.len()));
        let inner = entries.detach(py, || {
            let sgdd = CsrRef::new(&indptr, &indices, &data, n)
                .map_err(|e| error_to_py(&e))
                .and_then(|csr| Sgdd::reduce(csr, scaling).map_err(|e| error_to_py(&e)));
            // Freed before factoring: the copy never coexists with the factor's fill.
            drop((indptr, indices, data));
            sgdd.map(|sgdd| sgdd.factor(&config))
        })?;
        warn_on_scaling(py, &inner)?;
        Ok(PyFactor { inner, entries })
    }
}

/// Approximate Cholesky factorization for SDDM/Laplacian systems.
// GIL-free is sound: pyclass state is `Sync`, `Workspace` is borrow-checked, the module holds none, and `detach` reads only owned copies.
#[pymodule(gil_used = false)]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyFactor>()?;
    m.add_class::<PyScalingPolicy>()?;
    m.add_class::<PyWorkspace>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
