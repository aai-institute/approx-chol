use approx_chol::input::FactorWidth2;
use approx_chol::Real;

use super::lcg::Lcg;

/// A Gram cycle `x_i = u_{i+1} r_i`, `y_i = −u_i r_i` on `n` vertices, each value rounded to `T`.
pub struct Cycle<T> {
    n: usize,
    a: Vec<u32>,
    b: Vec<u32>,
    x: Vec<T>,
    y: Vec<T>,
}

impl<T: Real> Cycle<T> {
    /// `u = 2^min(i, n−1−i)` peaking mid-cycle and `r = 1`: every value exact.
    pub fn doubling(n: u32) -> Self {
        Self::new(n, |i| f64::from(1u32 << i.min(n - 1 - i)), |_| 1.0)
    }

    /// Random `u` and `r` in `[0.5, 1.5)`.
    pub fn random(n: u32, seed: u64) -> Self {
        let mut lcg = Lcg(seed);
        let mut draw = move || lcg.unit() + 0.5;
        let u: Vec<f64> = (0..n).map(|_| draw()).collect();
        let r: Vec<f64> = (0..n).map(|_| draw()).collect();
        Self::new(n, |i| u[i as usize], |i| r[i as usize])
    }

    fn new(n: u32, u: impl Fn(u32) -> f64, r: impl Fn(u32) -> f64) -> Self {
        let a: Vec<u32> = (0..n).collect();
        let b: Vec<u32> = (0..n).map(|i| (i + 1) % n).collect();
        let value = |x: f64| T::from(x).unwrap();
        let x = (a.iter().zip(&b))
            .map(|(&i, &j)| value(u(j) * r(i)))
            .collect();
        let y = a.iter().map(|&i| value(-u(i) * r(i))).collect();
        let n = n as usize;
        Self { n, a, b, x, y }
    }

    /// The closing column scaled by `1 + miss`, so the cycle has no null vector.
    pub fn missing(mut self, miss: f64) -> Self {
        let last = self.n - 1;
        let missed = self.x[last] * T::from(1.0 + miss).unwrap();
        assert!(missed != self.x[last], "{miss} rounds away in T");
        self.x[last] = missed;
        self
    }

    /// A column `√(weight·a_ii) e_i`, grounding vertex `i` by `weight·a_ii`.
    pub fn grounded(mut self, i: u32, weight: f64) -> Self {
        let at_i = |ends: &[u32], values: &[T]| -> f64 {
            (ends.iter().zip(values))
                .filter(|&(&end, _)| end == i)
                .map(|(_, &v)| (v * v).to_f64().unwrap())
                .sum()
        };
        let a_ii = at_i(&self.a, &self.x) + at_i(&self.b, &self.y);
        self.a.push(i);
        self.b.push((i + 1) % self.n as u32);
        self.x.push(T::from((weight * a_ii).sqrt()).unwrap());
        self.y.push(T::zero());
        self
    }

    /// Every column with its ends exchanged: the same matrix, a stated ground now on its `b` end.
    pub fn swapped(mut self) -> Self {
        std::mem::swap(&mut self.a, &mut self.b);
        std::mem::swap(&mut self.x, &mut self.y);
        self
    }

    pub fn gram(&self) -> FactorWidth2<'_, T, u32> {
        FactorWidth2::new(self.n, (&self.a, &self.x), (&self.b, &self.y)).expect("valid columns")
    }
}
