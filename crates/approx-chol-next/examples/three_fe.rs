//! Three fixed effects, no slopes: each pair of factors is one bipartite Gram, as within's Schwarz subdomains.

use approx_chol_next::input::{BipartiteGram, Incidence, Weights};
use approx_chol_next::{factorize, Config, Factor, ScalingPolicy, Sgdd};

/// Workers move between firms of their own region only, so (worker, firm) splits by region.
const REGIONS: [(usize, usize); 5] = [(2000, 200), (40, 6), (30, 5), (20, 4), (10, 3)];
const YEARS: usize = 10;

struct Panel {
    levels: [Vec<u32>; 3],
    n_levels: [usize; 3],
}

impl Panel {
    fn generate(seed: u64) -> Self {
        let mut rng = SplitMix(seed);
        let [mut workers, mut firms, mut years] = [vec![], vec![], vec![]];
        let (mut first_worker, mut first_firm) = (0, 0);
        for (n_workers, n_firms) in REGIONS {
            for worker in first_worker..first_worker + n_workers {
                for year in 0..YEARS {
                    workers.push(worker as u32);
                    firms.push((first_firm + rng.below(n_firms)) as u32);
                    years.push(year as u32);
                }
            }
            first_worker += n_workers;
            first_firm += n_firms;
        }
        Self {
            levels: [workers, firms, years],
            n_levels: [first_worker, first_firm, YEARS],
        }
    }
}

const NAMES: [&str; 3] = ["worker", "firm", "year"];

fn main() {
    let panel = Panel::generate(7);
    let incidences: Vec<Incidence<'_, f64>> = (0..3)
        .map(|f| {
            Incidence::new(
                panel.n_levels[f],
                &panel.levels[f],
                vec![None],
                Weights::Unit,
            )
        })
        .collect::<Result<_, _>>()
        .expect("valid incidence");
    for (l, r) in [(0, 1), (0, 2), (1, 2)] {
        println!("({}, {})", NAMES[l], NAMES[r]);
        let gram =
            BipartiteGram::new(incidences[l].side(0), incidences[r].side(0)).expect("valid gram");
        let sgdd = Sgdd::from_bipartite_gram(gram, ScalingPolicy::default()).expect("reduces");
        for component in sgdd.components() {
            let n = component.vertices().len();
            println!("  component: {n} vertices, floats {}", component.floats());
        }
        let a = PairGram {
            left: &panel.levels[l],
            right: &panel.levels[r],
            n_left: panel.n_levels[l],
        };
        let dense = Config {
            dense_up_to: usize::MAX,
            ..Config::default()
        };
        for (name, config) in [("dense", dense), ("default", Config::default())] {
            let factor = factorize(&sgdd, &config);
            println!("  {name}: fallbacks {}", factor.fallbacks().count());
            check(&a, &factor);
        }
    }
}

/// `A x` straight from the observations, independent of everything under test.
struct PairGram<'a> {
    left: &'a [u32],
    right: &'a [u32],
    n_left: usize,
}

impl PairGram<'_> {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len()];
        for (&a, &b) in self.left.iter().zip(self.right) {
            let (a, b) = (a as usize, self.n_left + b as usize);
            let t = x[a] + x[b];
            y[a] += t;
            y[b] += t;
        }
        y
    }
}

/// One application's residual, then PCG's iterations to `1e-10`; `b = A x_true` is consistent although `A` is singular.
fn check(a: &PairGram<'_>, factor: &Factor<f64>) {
    let mut rng = SplitMix(11);
    let x_true: Vec<f64> = (0..factor.n()).map(|_| rng.unit() - 0.5).collect();
    let b = a.apply(&x_true);
    let x = factor.solve(&b).expect("solves");
    println!(
        "  one solve: relative residual {:.2e}",
        relative_residual(a, &x, &b)
    );
    let (x, iterations) = pcg(a, factor, &b, 1e-10, 500);
    println!(
        "  pcg: {iterations} iterations, relative residual {:.2e}",
        relative_residual(a, &x, &b)
    );
}

fn relative_residual(a: &PairGram<'_>, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.apply(x);
    let r: Vec<f64> = b.iter().zip(&ax).map(|(b, ax)| b - ax).collect();
    norm(&r) / norm(b)
}

fn pcg(
    a: &PairGram<'_>,
    factor: &Factor<f64>,
    b: &[f64],
    tol: f64,
    max: usize,
) -> (Vec<f64>, usize) {
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut z = factor.solve(&r).expect("solves");
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for iteration in 0..max {
        if norm(&r) <= tol * norm(b) {
            return (x, iteration);
        }
        let ap = a.apply(&p);
        let alpha = rz / dot(&p, &ap);
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        z = factor.solve(&r).expect("solves");
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        p.iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
    }
    (x, max)
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    y.iter_mut().zip(x).for_each(|(y, x)| *y += alpha * x);
}

/// Deterministic and dependency-free, so the panel never moves under a `rand` upgrade.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
