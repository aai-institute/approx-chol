use approx_chol::input::Csr;
use approx_chol::{
    CliqueRule, Config, Factor, Multiplicity, ScalingPolicy, ScalingReport, Sgdd, SplitCount,
};
use numpy::{BorrowError, Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
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

/// An associated element type stops a call site pairing an index column with a float element.
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
    np.call_method1("ascontiguousarray", (cast,))?
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

#[pyclass(frozen, name = "Config")]
struct PyConfig(Config);

#[pymethods]
impl PyConfig {
    #[new]
    #[pyo3(signature = (seed=0, split=None, dense_up_to=24))]
    fn new(seed: u64, split: Option<u32>, dense_up_to: usize) -> PyResult<Self> {
        let approximate = match split {
            None => Multiplicity::Single,
            Some(k) => Multiplicity::Split(SplitCount::new(k).ok_or_else(|| {
                value_error(format!(
                    "split must be at least 2, got {k}; None selects AC"
                ))
            })?),
        };
        Ok(Self(Config {
            seed,
            complement: CliqueRule::Sampled(approximate),
            approximate,
            dense_up_to,
        }))
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed
    }

    #[getter]
    fn split(&self) -> Option<u32> {
        match self.0.approximate {
            Multiplicity::Single => None,
            Multiplicity::Split(k) => Some(k.get()),
        }
    }

    #[getter]
    fn dense_up_to(&self) -> usize {
        self.0.dense_up_to
    }
}

#[pyclass(frozen, name = "Factor")]
struct PyFactor {
    inner: Factor<f64, u32>,
}

/// Without this, a deficit the reduction clamped to make a block dominant is silent.
fn warn_on_scaling(py: Python<'_>, sgdd: &Sgdd<f64, u32>) -> PyResult<()> {
    let reports: Vec<ScalingReport> = (sgdd.blocks().iter())
        .filter_map(|block| block.scaling_report())
        .collect();
    if reports.is_empty() {
        return Ok(());
    }
    let worst = (reports.iter()).fold(0.0f64, |worst, report| worst.max(report.violation));
    let unsettled = reports.iter().filter(|report| !report.settled).count();
    let mut message = format!(
        "{} block(s) fell short of diagonal dominance by up to {worst:.3e} \
         of a diagonal; the deficit was clamped",
        reports.len()
    );
    if unsettled > 0 {
        message.push_str(&format!(
            "; {unsettled} block(s) did not settle, so a singular one may have been grounded"
        ));
    }
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (message, py.get_type::<pyo3::exceptions::PyRuntimeWarning>()),
    )?;
    Ok(())
}

fn factorize_csr(
    py: Python<'_>,
    csr: Csr<'_, f64, u32>,
    config: Option<&PyConfig>,
) -> PyResult<PyFactor> {
    let sgdd = Sgdd::from_csr(csr, ScalingPolicy::default()).map_err(|e| error_to_py(&e))?;
    warn_on_scaling(py, &sgdd)?;
    let config = config.map_or_else(Config::default, |config| config.0);
    let inner = approx_chol::factorize(&sgdd, &config);
    Ok(PyFactor { inner })
}

#[pymethods]
impl PyFactor {
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let n = self.inner.n();
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
        let x = self.inner.solve(b_slice).map_err(|e| error_to_py(&e))?;

        let mut out_rw = out.try_readwrite().map_err(|e| borrow_error("out", e))?;
        let out_slice = out_rw
            .as_slice_mut()
            .map_err(|_| value_error("out must be contiguous"))?;
        out_slice[..n].copy_from_slice(&x);
        Ok(())
    }
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

    let csr = Csr::new(n as usize, rp, ci, vals).map_err(|e| error_to_py(&e))?;
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
    if u32::try_from(shape.0).is_err() {
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

    let csr = Csr::new(shape.0, rp, ci, vals).map_err(|e| error_to_py(&e))?;
    factorize_csr(py, csr, config)
}

/// Approximate Cholesky factorization for SDDM/Laplacian systems.
// GIL-free is sound: every pyclass is `frozen` with `Sync` state and the module holds none.
#[pymodule(gil_used = false)]
fn _approx_chol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyFactor>()?;
    m.add_function(wrap_pyfunction!(factorize, m)?)?;
    m.add_function(wrap_pyfunction!(factorize_raw, m)?)?;
    Ok(())
}
