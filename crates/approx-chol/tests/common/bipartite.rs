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
