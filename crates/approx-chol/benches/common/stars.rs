use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

pub const STARS: usize = 10_000;

/// `STARS` stars, every one on the same `degree` kept vertices; only scales vary.
pub struct Stars {
    degree: usize,
    stars: Vec<u32>,
    kept: Vec<u32>,
    scales: Vec<f64>,
}

impl Stars {
    pub fn new(degree: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        let stars = (0..STARS * degree).map(|i| (i / degree) as u32).collect();
        let kept = (0..STARS * degree).map(|i| (i % degree) as u32).collect();
        let scales = (0..STARS * degree)
            .map(|_| rng.random_range(0.25..1.25f64).sqrt())
            .collect();
        Self {
            degree,
            stars,
            kept,
            scales,
        }
    }

    pub fn gram(&self) -> BipartiteFactorWidth2<'_, f64> {
        BipartiteFactorWidth2::scaled(
            FactorWidth1::new(STARS, &self.stars).expect("valid side"),
            FactorWidth1::new(self.degree, &self.kept).expect("valid side"),
            &self.scales,
        )
        .expect("valid gram")
    }
}
