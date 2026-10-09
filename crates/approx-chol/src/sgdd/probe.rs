//! Probe-only switches and counters for the scaling A/B; not part of the prototype's API.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Mutex;

/// The dominance scaling every later conversion runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// within's reduced CG.
    ReducedCg,
    /// SCI Algorithm 3.
    Sci,
    /// SCI Algorithm 3b.
    SciBalanced,
}

static METHOD: AtomicU8 = AtomicU8::new(0);

/// Selects the scaling for every later conversion in the process.
pub fn set_method(method: Method) {
    METHOD.store(method as u8, Ordering::Relaxed);
}

pub(crate) fn method() -> Method {
    match METHOD.load(Ordering::Relaxed) {
        1 => Method::Sci,
        2 => Method::SciBalanced,
        _ => Method::ReducedCg,
    }
}

/// One scaled component.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    /// Levels in the component.
    pub n: usize,
    /// Stored cells of its `C`.
    pub nnz: usize,
    /// Iterations the scaling spent.
    pub iterations: usize,
    /// Violation handed over.
    pub violation: f64,
    /// Wall time inside the scaling.
    pub seconds: f64,
}

static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

static RECORDING: AtomicBool = AtomicBool::new(false);

/// Times and records every later scaling; off by default, so the default path runs unobserved.
pub fn set_recording(on: bool) {
    RECORDING.store(on, Ordering::Relaxed);
}

pub(crate) fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

pub(crate) fn record(record: Record) {
    RECORDS.lock().expect("probe lock").push(record);
}

/// Drains the records gathered since the last call.
pub fn take() -> Vec<Record> {
    std::mem::take(&mut *RECORDS.lock().expect("probe lock"))
}
