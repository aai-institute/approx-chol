/// A fixed LCG, so drawn fixtures never move with a dependency.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = (self.0)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
