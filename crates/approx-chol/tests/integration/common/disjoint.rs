use approx_chol::input::FactorWidth2;

use super::lcg::Lcg;

/// A Gram `V Vᵀ` of disjoint components, one per entry of `sizes`, each a path plus as many drawn chords as vertices, its rows short of dominance; `V` has a null vector on the components `certified` names, elsewhere its cycles break one.
pub struct Disjoint {
    pub n: usize,
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Disjoint {
    pub fn new(sizes: &[usize], certified: impl Fn(usize) -> bool) -> Self {
        let mut lcg = Lcg(0x281);
        let (mut a, mut b, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        let mut first = 0;
        for (k, &size) in sizes.iter().enumerate() {
            let u: Vec<f64> = (0..size).map(|_| 0.5 + 1.5 * lcg.unit()).collect();
            let chords: Vec<(usize, usize)> = (0..size)
                .map(|_| {
                    (
                        lcg.next_u64() as usize % size,
                        lcg.next_u64() as usize % size,
                    )
                })
                .filter(|(i, j)| i != j)
                .collect();
            for (i, j) in (0..size - 1).map(|i| (i, i + 1)).chain(chords) {
                let x_k = 0.2 + 1.8 * lcg.unit();
                let y_k = match certified(k) {
                    true => -x_k * u[i] / u[j],
                    false => -0.2 - 1.8 * lcg.unit(),
                };
                a.push((first + i) as u32);
                b.push((first + j) as u32);
                x.push(x_k);
                y.push(y_k);
            }
            first += size;
        }
        Self {
            n: first,
            a,
            b,
            x,
            y,
        }
    }

    pub fn view(&self) -> FactorWidth2<'_, f64> {
        FactorWidth2::new(self.n, (&self.a, &self.x), (&self.b, &self.y)).expect("valid gram")
    }
}
