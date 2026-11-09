use super::bipartite::Bipartite;

/// A frustrated unit cycle `L0 R0 L1 R1 …`, `cycle` vertices a side, and `pendants` left vertices each joined by `1e-95` to `wide` cycle vertices and left a `1e-200` surplus: within the band.
pub struct Cycle {
    pub cycle: usize,
    pub pendants: usize,
    pub wide: usize,
    /// Unit pendants on the right, which make the left side, surplus and all, the kept one.
    pub right_pendants: usize,
}

impl Cycle {
    /// The cycle and its left pendants as a bipartite `V`, its right pendants after the cycle on the right.
    pub fn bipartite(&self) -> Bipartite {
        let m = self.cycle;
        let (mut left, mut right, mut x, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut push = |l: usize, r: usize, a: f64, b: f64| {
            left.push(l as u32);
            right.push(r as u32);
            x.push(a);
            y.push(b);
        };
        for i in 0..m {
            push(i, i, 1.0, 1.0);
            push(i, (i + 1) % m, 1.0, if i == m - 1 { -1.0 } else { 1.0 });
        }
        for p in 0..self.pendants {
            for k in 0..self.wide {
                let y = if k == 0 { -1e-95 } else { 1e-95 };
                push(m + p, (p + k) % m, 1e-95, y);
            }
            push(m + p, 0, 1e-100, 0.0);
        }
        for p in 0..self.right_pendants {
            push(p % m, m + p, 1.0, 1.0);
        }
        Bipartite {
            n_left: m + self.pendants,
            n_right: m + self.right_pendants,
            left,
            right,
            values: Some((x, y)),
        }
    }
}
