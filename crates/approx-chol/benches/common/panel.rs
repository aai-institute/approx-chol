use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use rand::rngs::SmallRng;
use rand::RngExt;

const OBSERVATIONS_PER_WORKER: usize = 5;

/// A worker-firm panel as `within` passes it: each worker observed at firms drawn uniformly.
pub struct Panel {
    workers: usize,
    firms: usize,
    worker: Vec<u32>,
    firm: Vec<u32>,
}

impl Panel {
    pub fn new(workers: usize, firms: usize, rng: &mut SmallRng) -> Self {
        let worker: Vec<u32> = (0..workers * OBSERVATIONS_PER_WORKER)
            .map(|i| (i / OBSERVATIONS_PER_WORKER) as u32)
            .collect();
        let firm: Vec<u32> = (0..worker.len())
            .map(|_| rng.random_range(0..firms as u32))
            .collect();
        Self {
            workers,
            firms,
            worker,
            firm,
        }
    }

    /// The panel's Gram, workers on the left.
    pub fn gram(&self) -> BipartiteFactorWidth2<'_, f64> {
        BipartiteFactorWidth2::new(
            FactorWidth1::new(self.workers, &self.worker, None).expect("valid side"),
            FactorWidth1::new(self.firms, &self.firm, None).expect("valid side"),
        )
        .expect("valid gram")
    }
}
