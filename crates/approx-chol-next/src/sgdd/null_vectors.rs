use crate::input::{Column, IntoColumns};
use crate::storage::widen;
use crate::Real;

/// Per vertex, a `u` orthogonal to its component's coupling columns, `NaN` where none is.
pub(super) struct NullVectors {
    u: Vec<f64>,
}

impl NullVectors {
    /// One union-find pass; a cycle off by more than `√ε` relative dooms its component.
    pub(super) fn of<'c, T: Real>(
        v: impl IntoColumns<T>,
        components: impl Iterator<Item = &'c [u32]>,
    ) -> Self {
        let n = v.n();
        let mut component = vec![0u32; n];
        // A lone vertex is dead or definite, never a candidate.
        let mut doomed = Vec::new();
        for (c, vertices) in components.enumerate() {
            vertices
                .iter()
                .for_each(|&v| component[v as usize] = c as u32);
            doomed.push(vertices.len() == 1);
        }
        let mut alive = doomed.iter().filter(|&&doomed| !doomed).count();
        let mut forest = RatioForest::new(n, widen(T::epsilon()).sqrt());
        v.for_each_column(
            #[inline(always)]
            |column| {
                if alive == 0 {
                    return;
                }
                let Some(c) = coupling(&column, &component) else {
                    return;
                };
                let k = -widen(column.v_a) / widen(column.v_b);
                if !doomed[c] && !forest.join(column.a as usize, column.b as usize, k) {
                    doomed[c] = true;
                    alive -= 1;
                }
            },
        );
        let u = (0..n)
            .map(|i| match doomed[component[i] as usize] {
                true => f64::NAN,
                false => forest.find(i).1,
            })
            .collect();
        Self { u }
    }

    /// The component's `u`, unless it was doomed.
    pub(super) fn on(&self, vertices: &[u32]) -> Option<Vec<f64>> {
        let u: Vec<f64> = vertices.iter().map(|&v| self.u[v as usize]).collect();
        u.iter().all(|x| x.is_finite()).then_some(u)
    }
}

/// A column's component when both ends are nonzero in it.
fn coupling<T: Real>(column: &Column<T>, component: &[u32]) -> Option<usize> {
    let c = component[column.a as usize];
    let couples = !column.weight.is_zero()
        && column.v_a != T::zero()
        && column.v_b != T::zero()
        && component[column.b as usize] == c;
    couples.then_some(c as usize)
}

/// Union-find whose links carry `u_child / u_parent`.
struct RatioForest {
    parent: Vec<u32>,
    /// `1` at a root.
    ratio: Vec<f64>,
    size: Vec<u32>,
    /// How far a closing cycle may miss, relative.
    tolerance: f64,
}

impl RatioForest {
    fn new(n: usize, tolerance: f64) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            ratio: vec![1.0; n],
            size: vec![1; n],
            tolerance,
        }
    }

    /// Root and `u_i / u_root`, halving the path.
    fn find(&mut self, mut i: usize) -> (usize, f64) {
        let mut x = 1.0;
        while self.parent[i] as usize != i {
            let up = self.parent[i] as usize;
            self.ratio[i] *= self.ratio[up];
            self.parent[i] = self.parent[up];
            x *= self.ratio[i];
            i = self.parent[i] as usize;
        }
        (i, x)
    }

    /// Joins `u_b = k·u_a`; `false` on a broken cycle or a ratio leaving the finite nonzero range.
    fn join(&mut self, a: usize, b: usize, k: f64) -> bool {
        // An overflowed `v_a/v_b` would compare `∞ ≤ tolerance·∞` and pass.
        if !k.is_finite() {
            return false;
        }
        let ((ra, xa), (rb, xb)) = (self.find(a), self.find(b));
        if ra == rb {
            let (want, have) = (k * xa, xb);
            return (want - have).abs() <= self.tolerance * want.abs().max(have.abs());
        }
        // `u_rb = (k·xa/xb)·u_ra`, inverted when `ra` goes under `rb`.
        let (big, small, ratio) = match self.size[ra] >= self.size[rb] {
            true => (ra, rb, k * xa / xb),
            false => (rb, ra, xb / (k * xa)),
        };
        self.parent[small] = big as u32;
        self.ratio[small] = ratio;
        self.size[big] += self.size[small];
        ratio.is_finite() && ratio != 0.0
    }
}
