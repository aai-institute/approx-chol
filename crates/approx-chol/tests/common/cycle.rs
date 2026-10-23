use approx_chol::input::FactorWidth2;

/// A Gram cycle on `n` vertices whose null vector `u = 2^min(i, n−1−i)` peaks mid-cycle, its closing column off by `1 + miss`.
pub struct Cycle {
    a: Vec<u32>,
    b: Vec<u32>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Cycle {
    pub fn new(n: u32, miss: f64) -> Self {
        let u = |i: u32| f64::from(1u32 << i.min(n - 1 - i));
        let a: Vec<u32> = (0..n).collect();
        let b: Vec<u32> = (0..n).map(|i| (i + 1) % n).collect();
        let mut x: Vec<f64> = b.iter().map(|&j| u(j)).collect();
        let y: Vec<f64> = a.iter().map(|&i| -u(i)).collect();
        x[n as usize - 1] *= 1.0 + miss;
        Self { a, b, x, y }
    }

    pub fn gram(&self) -> FactorWidth2<'_, f64, u32> {
        FactorWidth2::new(self.a.len(), (&self.a, &self.x), (&self.b, &self.y))
            .expect("valid columns")
    }
}
