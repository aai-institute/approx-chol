use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};

/// A bipartite `V`'s columns `(left, right, x, y)`; no `x` and `y` reads as an incidence.
pub struct Bipartite {
    pub n_left: usize,
    pub n_right: usize,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub values: Option<(Vec<f64>, Vec<f64>)>,
}

impl Bipartite {
    /// Three components of `(left, right, frustrated)` vertices, 48 entries to a left vertex: the first's 3200·48 entries clear the `Cᵀ` gate (2¹⁷) by 17%, so its elimination's solves split; each splits its star fold and exact rows, together they split the components.
    pub fn three_groups() -> Self {
        const DEGREE: usize = 48;
        let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        let (mut first_left, mut first_right) = (0, 0);
        for (n_left, n_right, frustrated) in
            [(3200, 700, false), (420, 1000, true), (420, 1000, false)]
        {
            for i in 0..n_left {
                for d in 0..DEGREE {
                    let k = left.len();
                    left.push((first_left + i) as u32);
                    right.push((first_right + (i * 7 + d * 13) % n_right) as u32);
                    x.push(1.0 + (k % 7) as f64 / 8.0);
                    let magnitude = 0.5 + (k % 3) as f64 / 4.0;
                    let sign = if frustrated && k % 5 == 0 { 1.0 } else { -1.0 };
                    y.push(sign * magnitude);
                }
            }
            (first_left, first_right) = (first_left + n_left, first_right + n_right);
        }
        Self {
            n_left: first_left,
            n_right: first_right,
            left,
            right,
            values: Some((x, y)),
        }
    }

    pub fn view(&self) -> BipartiteFactorWidth2<'_, f64> {
        let (x, y) = match &self.values {
            Some((x, y)) => (Some(&x[..]), Some(&y[..])),
            None => (None, None),
        };
        BipartiteFactorWidth2::new(
            FactorWidth1::new(self.n_left, &self.left, x).expect("valid left"),
            FactorWidth1::new(self.n_right, &self.right, y).expect("valid right"),
        )
        .expect("valid gram")
    }
}
