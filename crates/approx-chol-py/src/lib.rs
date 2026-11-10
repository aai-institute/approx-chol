use approx_chol::input::{CsrRef, InputError};
use approx_chol::{
    ApproxInverse, Config, CoverSurplus, FrustratedRoute, Header, MatrixMut, Sampling,
    ScalingOutcome, ScalingPolicy, Sgdd, Threads,
};
use num_traits::PrimInt;
use numpy::ndarray::{ArrayRef, Dimension, Ix1, Ix2};
use numpy::{
    BorrowError, Element, PyArray, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadwriteArray, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyNotImplementedError, PyOverflowError, PyTypeError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyTuple, PyType};
use std::num::NonZeroU64;
use std::sync::Arc;

#[inline]
fn value_error(message: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.to_string())
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

/// An array's role, which fixes the dtype kinds it may hold without truncation.
#[derive(Clone, Copy)]
enum Column {
    Index,
    Value,
}

impl Column {
    fn accepts(self, kind: u8) -> bool {
        match self {
            Self::Index => matches!(kind, b'i' | b'u'),
            Self::Value => matches!(kind, b'i' | b'u' | b'f'),
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
    let array = arr.cast::<PyUntypedArray>()?;
    if array.ndim() != 1 {
        return Err(value_error(format!("{name} must be a 1-D array")));
    }
    if !column.accepts(array.dtype().kind()) {
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

/// Whether rust-numpy reads the array right: it neither checks alignment nor strides that split an `f64`.
fn viewable(array: &Bound<'_, PyUntypedArray>) -> bool {
    let whole = |stride: &isize| stride.unsigned_abs() % size_of::<f64>() == 0;
    array.is_aligned() && array.strides().iter().all(whole)
}

/// `b` as a new `f64` array of its shape and layout, the block an apply solves and returns.
fn block<'py, D: Dimension>(
    b: &Bound<'py, PyUntypedArray>,
) -> PyResult<Bound<'py, PyArray<f64, D>>> {
    if !Column::Value.accepts(b.dtype().kind()) {
        let dtype = b.dtype();
        return Err(PyTypeError::new_err(format!(
            "b must have a real dtype, not {dtype}"
        )));
    }
    match b.cast::<PyArray<f64, D>>() {
        // Rust's copy keeps the GIL under #460's bound, numpy's releases it; ndarray rejects an empty view's free strides.
        Ok(f64s) if viewable(b) && !b.is_empty() => {
            let f64s = f64s.try_readonly().map_err(|e| borrow_error("b", e))?;
            Ok(PyArray::from_owned_array(
                b.py(),
                f64s.as_array().to_owned(),
            ))
        }
        _ => (b.call_method1("astype", (f64::get_dtype(b.py()),))?)
            .cast_into()
            .map_err(Into::into),
    }
}

/// ndarray's rule for a writable view, unchecked in release: each axis, by stride, steps past the span of all faster ones.
fn entries_overlap(shape: &[usize], strides: &[isize]) -> bool {
    if shape.contains(&0) {
        return false;
    }
    let mut axes: Vec<(usize, usize)> = (strides.iter().zip(shape))
        .filter(|&(_, &extent)| extent > 1)
        .map(|(stride, &extent)| (stride.unsigned_abs() / size_of::<f64>(), extent))
        .collect();
    axes.sort_unstable();
    let mut span = 0;
    axes.into_iter().any(|(stride, extent)| {
        let overlaps = stride <= span;
        span += (extent - 1) * stride;
        overlaps
    })
}

/// `out` borrowed to receive `x`: a writable, aligned `f64` array of `x`'s shape whose entries neither overlap nor interleave.
fn destination<'py, D: Dimension>(
    out: &Bound<'py, PyAny>,
    x: &Bound<'py, PyArray<f64, D>>,
) -> PyResult<PyReadwriteArray<'py, f64, D>> {
    let out = (out.cast::<PyArrayDyn<f64>>())
        .map_err(|_| PyTypeError::new_err("out must be a float64 array"))?;
    if out.shape() != x.shape() {
        let (found, expected) = (out.getattr("shape")?, x.getattr("shape")?);
        return Err(value_error(format!(
            "out has shape {found}, expected {expected}"
        )));
    }
    let out = out.cast::<PyArray<f64, D>>()?;
    if !viewable(out.as_untyped()) {
        return Err(value_error(
            "out must be aligned, its strides whole float64s",
        ));
    }
    if entries_overlap(out.shape(), out.strides()) {
        return Err(value_error("out's entries overlap or interleave"));
    }
    out.try_readwrite().map_err(|e| borrow_error("out", e))
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

#[pyclass(frozen, eq, hash, name = "Config")]
#[derive(PartialEq, Eq, Hash)]
struct PyConfig(Config);

/// Each frustrated route by its Python name.
const FRUSTRATED: [(&str, FrustratedRoute); 4] = [
    ("cover", FrustratedRoute::Cover(CoverSurplus::Auto)),
    (
        "cover_ground_vertex",
        FrustratedRoute::Cover(CoverSurplus::GroundVertex),
    ),
    (
        "cover_surplus_edges",
        FrustratedRoute::Cover(CoverSurplus::SurplusEdges),
    ),
    ("signed", FrustratedRoute::Signed),
];

/// The route `name` names, else a `ValueError` listing every name.
fn frustrated_route(name: &str) -> PyResult<FrustratedRoute> {
    let named = FRUSTRATED.iter().find(|(n, _)| *n == name);
    named.map(|&(_, route)| route).ok_or_else(|| {
        let names = FRUSTRATED.map(|(n, _)| format!("'{n}'")).join(", ");
        value_error(format!("frustrated must be one of {names}, not '{name}'"))
    })
}

/// `value` as a positive `N`: an integer past `N` or below 1 is a `ValueError`, a non-integer or a bool a `TypeError`.
fn positive<N: TryFrom<NonZeroU64>>(value: &Bound<'_, PyAny>, name: &str) -> PyResult<N> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be an integer, not a bool"
        )));
    }
    let bits = 8 * size_of::<N>();
    let out_of_range = || value_error(format!("{name} must be an integer in [1, 2**{bits} - 1]"));
    match value.extract::<u64>() {
        Ok(k) => (NonZeroU64::new(k))
            .and_then(|k| N::try_from(k).ok())
            .ok_or_else(out_of_range),
        Err(e) if e.is_instance_of::<PyOverflowError>(value.py()) => Err(out_of_range()),
        Err(e) => Err(e),
    }
}

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=None, split=None, exact_max_dim=None, *, frustrated=None))]
    fn new(
        seed: Option<u64>,
        split: Option<Bound<'_, PyAny>>,
        exact_max_dim: Option<usize>,
        frustrated: Option<&str>,
    ) -> PyResult<Self> {
        let default = Config::default();
        let sampling = match split {
            None => default.sampling,
            Some(split) => Sampling::CliqueTree {
                copies: positive(&split, "split")?,
            },
        };
        Ok(Self(Config {
            seed: seed.unwrap_or(default.seed),
            sampling,
            frustrated: frustrated.map_or(Ok(default.frustrated), frustrated_route)?,
            exact_max_dim: exact_max_dim.unwrap_or(default.exact_max_dim),
            ..default
        }))
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed
    }

    #[getter]
    fn split(&self) -> PyResult<u32> {
        match self.0.sampling {
            Sampling::CliqueTree { copies } => Ok(copies.get()),
            _ => Err(PyNotImplementedError::new_err("this sampling has no split")),
        }
    }

    #[getter]
    fn frustrated(&self) -> PyResult<&'static str> {
        let named = FRUSTRATED
            .iter()
            .find(|(_, route)| *route == self.0.frustrated);
        named
            .map(|&(name, _)| name)
            .ok_or_else(|| PyNotImplementedError::new_err("this frustrated route has no name"))
    }

    #[getter]
    fn exact_max_dim(&self) -> usize {
        self.0.exact_max_dim
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

/// A build's or an apply's budget: `None` every core, else a count.
fn threads(threads: Option<&Bound<'_, PyAny>>) -> PyResult<Threads> {
    threads.map_or(Ok(Threads::All), |threads| {
        positive(threads, "threads").map(Threads::Count)
    })
}

#[pyclass(frozen, module = "approx_chol", name = "Clamp")]
struct PyClamp {
    #[pyo3(get)]
    violation: f64,
    #[pyo3(get)]
    scaling: &'static str,
}

#[pyclass(frozen, module = "approx_chol", name = "Component")]
struct PyComponent {
    #[pyo3(get)]
    vertices: Py<PyArray1<isize>>,
    #[pyo3(get)]
    clamp: Option<Py<PyClamp>>,
}

/// What every budget's operator over one build shares.
struct Factor {
    inner: ApproxInverse<f64>,
    entries: StoredEntries,
    components: PyOnceLock<Py<PyTuple>>,
}

#[pyclass(frozen, module = "approx_chol._approx_chol", name = "_ApproxInverse")]
struct PyApproxInverse {
    factor: Arc<Factor>,
    threads: Threads,
}

impl PyApproxInverse {
    fn new(inner: ApproxInverse<f64>, entries: StoredEntries, threads: Threads) -> Self {
        let components = PyOnceLock::new();
        let factor = Arc::new(Factor {
            inner,
            entries,
            components,
        });
        Self { factor, threads }
    }

    /// `b` solved at the held budget, into `out` when given.
    fn apply_as<'py, D: Dimension>(
        &self,
        b: &Bound<'py, PyUntypedArray>,
        out: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>>
    where
        ArrayRef<f64, D>: MatrixMut<f64>,
    {
        let x = block::<D>(b)?;
        let written = out.map(|out| destination(out, &x)).transpose()?;
        {
            let mut solution = x.try_readwrite().map_err(|e| borrow_error("b", e))?;
            let mut solution = solution.as_array_mut();
            let Factor { inner, entries, .. } = &*self.factor;
            let solved = entries.detach(b.py(), || inner.apply(&mut *solution, self.threads));
            solved.expect("b has n rows");
        }
        match written {
            None => Ok(x.into_any()),
            Some(mut written) => {
                if !x.is_empty() {
                    written.as_array_mut().assign(&x.readonly().as_array());
                }
                Ok(written.as_any().clone())
            }
        }
    }
}

/// Without this, a deficit the reduction clamped to make a block dominant is silent.
fn warn_on_scaling(py: Python<'_>, factor: &ApproxInverse<f64>) -> PyResult<()> {
    let (mut clamped, mut worst, mut unsettled) = (0, 0.0f64, 0);
    for component in factor.components() {
        if let Some(clamp) = component.clamp() {
            clamped += 1;
            worst = worst.max(clamp.violation);
        }
        unsettled += usize::from(component.grounding_unproven());
    }
    let mut causes = Vec::new();
    if clamped > 0 {
        causes.push(format!(
            "{clamped} block(s) fell short of diagonal dominance by up to {worst:.3e} \
             of a diagonal; the deficit was clamped"
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
impl PyApproxInverse {
    /// The factor's saved form, postcard-encoded, and nothing else.
    fn _saved_form<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let form = postcard::to_allocvec(&self.factor.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &form))
    }

    /// A failed decode reads the header apart, as postcard drops the message naming a mismatch.
    #[classmethod]
    fn _load(_cls: &Bound<'_, PyType>, form: &[u8]) -> PyResult<Self> {
        let invalid = |reason: &dyn std::fmt::Display| {
            value_error(format!("the saved ApproxInverse is invalid: {reason}"))
        };
        match postcard::take_from_bytes::<ApproxInverse<f64>>(form) {
            // Interim: the form's length stands in for the input's entries until #460 settles the bound's owner.
            Ok((inner, [])) => Ok(Self::new(
                inner,
                StoredEntries(form.len() / size_of::<f64>()),
                Threads::All,
            )),
            Ok((_, rest)) => Err(invalid(&format_args!("{} trailing bytes", rest.len()))),
            Err(error) => {
                let mismatch = postcard::take_from_bytes::<Header>(form)
                    .ok()
                    .and_then(|(header, _)| header.expect::<f64>().err());
                Err(mismatch.map_or_else(|| invalid(&error), value_error))
            }
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.factor.inner.n()
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.factor.inner.n_steps()
    }

    #[getter]
    fn threads(&self) -> Option<usize> {
        match self.threads {
            Threads::All => None,
            Threads::Count(count) => Some(count.get()),
        }
    }

    fn with_threads(&self, threads: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let factor = Arc::clone(&self.factor);
        Ok(Self {
            factor,
            threads: self::threads(threads)?,
        })
    }

    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<&Py<PyTuple>> {
        self.factor.components.get_or_try_init(py, || {
            let components = self.factor.inner.components().map(|component| {
                let vertices = component
                    .vertices()
                    .map(|v| isize::try_from(v).expect("an index"));
                let vertices = PyArray1::from_iter(py, vertices);
                vertices.readwrite().make_nonwriteable();
                let clamp = component.clamp().map(|clamp| PyClamp {
                    violation: clamp.violation,
                    scaling: match clamp.scaling {
                        ScalingOutcome::Certified => "certified",
                        ScalingOutcome::Refuted => "refuted",
                        ScalingOutcome::Unsettled => "unsettled",
                    },
                });
                let clamp = clamp.map(|clamp| Py::new(py, clamp)).transpose()?;
                let vertices = vertices.unbind();
                Py::new(py, PyComponent { vertices, clamp })
            });
            let components = components.collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, components).map(Bound::unbind)
        })
    }

    /// Solves `b` into a new array, or into `out`, which may be `b`: the solve runs on its own copy, as without the GIL another thread may touch `out`.
    fn apply<'py>(
        &self,
        b: &Bound<'py, PyAny>,
        out: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let b = match b.cast::<PyUntypedArray>() {
            Ok(b) => b.clone(),
            Err(_) => (b.py().import("numpy")?)
                .call_method1("asarray", (b,))?
                .cast_into()?,
        };
        let n = self.factor.inner.n();
        match b.shape() {
            [rows] if *rows == n => self.apply_as::<Ix1>(&b, out),
            [rows, _] if *rows == n => self.apply_as::<Ix2>(&b, out),
            _ => {
                let shape = b.getattr("shape")?;
                Err(value_error(format!(
                    "b has shape {shape}, expected ({n},) or ({n}, k)"
                )))
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, n, config, scaling))]
fn factorize_raw<'py>(
    py: Python<'py>,
    row_ptrs: &Bound<'py, PyAny>,
    col_indices: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    n: usize,
    config: Option<&PyConfig>,
    scaling: Option<&PyScalingPolicy>,
) -> PyResult<PyApproxInverse> {
    let np = py.import("numpy")?;
    CsrArrays {
        n,
        indptr: as_1d(&np, row_ptrs, "row_ptrs", Column::Index)?,
        indices: as_1d(&np, col_indices, "col_indices", Column::Index)?,
        data: as_contiguous(&np, &as_1d(&np, values, "values", Column::Value)?)?,
        np,
    }
    .factorize(config, scaling, Threads::All)
}

#[pyfunction]
#[pyo3(signature = (matrix, config, scaling, threads))]
fn factorize(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    config: Option<&PyConfig>,
    scaling: Option<&PyScalingPolicy>,
    threads: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyApproxInverse> {
    let threads = self::threads(threads)?;
    let (n, columns) = matrix.getattr("shape")?.extract::<(usize, usize)>()?;
    if n != columns {
        let error = InputError::NotSquare {
            rows: n,
            cols: columns,
        };
        return Err(value_error(error));
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
    .factorize(config, scaling, threads)
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
        threads: Threads,
    ) -> PyResult<PyApproxInverse> {
        let dtypes = (
            self.indptr.getattr("dtype")?,
            self.indices.getattr("dtype")?,
        );
        let dtype = self.np.call_method1("result_type", dtypes)?;
        let kind = dtype.getattr("kind")?.extract::<String>()?;
        let width = dtype.getattr("itemsize")?.extract::<usize>()?;
        match (kind.as_str(), width) {
            ("i", 4) => self.factorize_as::<i32>(config, scaling, threads),
            ("u", 4) => self.factorize_as::<u32>(config, scaling, threads),
            ("u", 8) => self.factorize_as::<u64>(config, scaling, threads),
            // Narrower integers widen; int64 beside uint64 promotes to float, and i64 holds any valid index.
            _ => self.factorize_as::<i64>(config, scaling, threads),
        }
    }

    /// Copies the arrays, so nothing after reads a caller's array.
    fn factorize_as<J: Element + PrimInt + Sync + Send>(
        &self,
        config: Option<&PyConfig>,
        scaling: Option<&PyScalingPolicy>,
        threads: Threads,
    ) -> PyResult<PyApproxInverse> {
        let indptr = as_contiguous::<J>(&self.np, &self.indptr)?;
        let indices = as_contiguous::<J>(&self.np, &self.indices)?;
        let indptr = owned(&indptr, "indptr")?;
        let indices = owned(&indices, "indices")?;
        let data = owned(&self.data, "data")?;
        let config = config.map_or_else(Config::default, |config| config.0);
        let scaling = scaling.map_or_else(ScalingPolicy::default, |scaling| scaling.0);
        let (py, n, entries) = (self.np.py(), self.n, StoredEntries(data.len()));
        let inner = entries.detach(py, || {
            let sgdd = CsrRef::new(n, &indptr, &indices, &data)
                .map_err(value_error)
                .and_then(|csr| Sgdd::reduce(csr, scaling, threads).map_err(value_error));
            // Freed before factoring: the copy never coexists with the factor's fill.
            drop((indptr, indices, data));
            sgdd.map(|sgdd| sgdd.factor(config, threads))
        })?;
        warn_on_scaling(py, &inner)?;
        Ok(PyApproxInverse::new(inner, entries, threads))
    }
}

/// Approximate Cholesky factorization for SDDM/Laplacian systems.
// GIL-free is sound: pyclass state is `Sync` and immutable, the module holds none, and `detach` touches only owned copies.
#[pymodule(gil_used = false)]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyApproxInverse>()?;
    m.add_class::<PyClamp>()?;
    m.add_class::<PyComponent>()?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyScalingPolicy>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
