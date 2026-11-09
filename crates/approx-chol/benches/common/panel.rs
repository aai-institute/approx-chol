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
    /// Each observation's firm slope, its sign either way; `None` is the firm intercept.
    covariate: Option<Vec<f64>>,
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
            covariate: None,
        }
    }

    /// The firm column scaled by a covariate of magnitude in `[0.25, 2)` and either sign: a frustrated Gram.
    #[allow(dead_code, reason = "only the frustrated benches draw slopes")]
    pub fn with_slope(workers: usize, firms: usize, rng: &mut SmallRng) -> Self {
        let mut panel = Self::new(workers, firms, rng);
        let slope = |_| match rng.random_range(0.25..2.0) {
            magnitude if rng.random_bool(0.5) => magnitude,
            magnitude => -magnitude,
        };
        panel.covariate = Some((0..panel.worker.len()).map(slope).collect());
        panel
    }

    /// The panel's Gram, workers on the left.
    pub fn gram(&self) -> BipartiteFactorWidth2<'_, f64> {
        BipartiteFactorWidth2::new(
            FactorWidth1::new(self.workers, &self.worker, None).expect("valid side"),
            FactorWidth1::new(self.firms, &self.firm, self.covariate.as_deref())
                .expect("valid side"),
        )
        .expect("valid gram")
    }

    /// Each observation `(worker, firm, value)`, firms numbered after workers.
    #[allow(
        dead_code,
        reason = "only the frustrated benches read the observations"
    )]
    pub fn observations(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.worker.len()).map(|k| {
            let value = self.covariate.as_ref().map_or(1.0, |c| c[k]);
            let firm = self.workers + self.firm[k] as usize;
            (self.worker[k] as usize, firm, value)
        })
    }

    #[allow(dead_code, reason = "only the frustrated benches read the dimension")]
    pub fn n(&self) -> usize {
        self.workers + self.firms
    }
}
