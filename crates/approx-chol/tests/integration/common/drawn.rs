use super::bipartite::Bipartite;
use super::lcg::Lcg;

impl Bipartite {
    /// `columns` columns over sides of `n_left` and `n_right`, left vertex `n_left − 1` in none, so it is dead.
    pub fn new(n_left: usize, n_right: usize, columns: usize, signed: bool) -> Self {
        let mut lcg = Lcg(0x9e37_79b9);
        let mut next = || (lcg.next_u64() >> 33) as usize;
        let left: Vec<u32> = (0..columns).map(|k| (k % (n_left - 1)) as u32).collect();
        let right: Vec<u32> = (0..columns).map(|_| (next() % n_right) as u32).collect();
        let values = signed.then(|| {
            let mut magnitude = || 0.5 + (next() % 1000) as f64 / 1000.0;
            let x: Vec<f64> = (0..columns).map(|_| magnitude()).collect();
            let y: Vec<f64> = (0..columns)
                .map(|k| magnitude() * if k % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            (x, y)
        });
        Self {
            n_left,
            n_right,
            left,
            right,
            values,
        }
    }
}
