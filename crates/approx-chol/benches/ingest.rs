#[path = "common/csr.rs"]
mod csr;
#[path = "common/grid.rs"]
mod grid;
#[path = "common/panel.rs"]
mod panel;
#[path = "common/powerlaw.rs"]
mod powerlaw;
#[path = "common/stars.rs"]
mod stars;

use std::hint::black_box;

use approx_chol::input::{Input, Laplacian, Sddm};
use approx_chol::Config;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use csr::{build, OwnedCsr};
use grid::grid_laplacian;
use panel::Panel;
use powerlaw::barabasi_albert;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use stars::Stars;

/// The diagonal and the entries right of it.
fn upper(csr: &OwnedCsr) -> OwnedCsr {
    OwnedCsr::from_rows((0..csr.n()).map(|i| csr.row(i).filter(move |&(j, _)| j as usize >= i)))
}

/// Every row's entries in descending column order, which construction and factorize each rewrite.
fn reversed(csr: &OwnedCsr) -> OwnedCsr {
    OwnedCsr::from_rows((0..csr.n()).map(|i| {
        let mut entries: Vec<_> = csr.row(i).collect();
        entries.reverse();
        entries
    }))
}

/// A Laplacian's edges `(u, v, w)`, `u < v`, read from its upper triangle.
struct Edges {
    n: usize,
    u: Vec<u32>,
    v: Vec<u32>,
    w: Vec<f64>,
    surplus: Vec<f64>,
}

impl Edges {
    fn of(csr: &OwnedCsr) -> Self {
        let (mut u, mut v, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..csr.n() {
            for (j, a) in csr.row(i).filter(|&(j, _)| j as usize > i) {
                u.push(i as u32);
                v.push(j);
                w.push(-a);
            }
        }
        // Grounded at one vertex, so the SDDM differs from the Laplacian by one diagonal.
        let mut surplus = vec![0.0; csr.n()];
        surplus[0] = 1.0;
        Self {
            n: csr.n(),
            u,
            v,
            w,
            surplus,
        }
    }

    fn laplacian(&self) -> Laplacian<'_, f64> {
        Laplacian::weighted(self.n, &self.u, &self.v, &self.w).expect("valid edges")
    }

    fn sddm(&self) -> Sddm<'_, f64> {
        Sddm::new(self.laplacian(), &self.surplus).expect("valid surplus")
    }
}

/// `build` times construction and factorize together, `construct` and `factorize` each alone.
fn ingest<'a, V>(group: &mut BenchmarkGroup<'_, WallTime>, case: &str, construct: impl Fn() -> V)
where
    V: Copy + Into<Input<'a, f64, u32>>,
{
    let factorize = |view: V| build(view, Config::default());
    group.bench_function(BenchmarkId::new("build", case), |b| {
        b.iter(|| factorize(construct()))
    });
    group.bench_function(BenchmarkId::new("construct", case), |b| {
        b.iter(|| black_box(construct()))
    });
    let view = construct();
    group.bench_function(BenchmarkId::new("factorize", case), |b| {
        b.iter(|| factorize(black_box(view)))
    });
}

fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    group.sample_size(10);

    // 1000x1000 holds 5M nonzeros, so ingestion runs out of cache.
    let grid = grid_laplacian(1000, 1000);
    ingest(&mut group, "grid1000_csr", || grid.csr());
    let grid_upper = upper(&grid);
    ingest(&mut group, "grid1000_upper", || grid_upper.upper());
    let grid_reversed = reversed(&grid);
    ingest(&mut group, "grid1000_reversed", || grid_reversed.csr());
    let grid_edges = Edges::of(&grid);
    ingest(&mut group, "grid1000_laplacian", || grid_edges.laplacian());
    ingest(&mut group, "grid1000_sddm", || grid_edges.sddm());

    let powerlaw = barabasi_albert(20_000, 5, 0xDEAD);
    ingest(&mut group, "powerlaw_20k_m5", || powerlaw.csr());

    let panel = Panel::new(200_000, 10_000, &mut SmallRng::seed_from_u64(0x5EED));
    ingest(&mut group, "panel_200000x10000", || panel.gram());

    let stars = Stars::new(10);
    ingest(&mut group, "stars_10", || stars.gram());

    group.finish();
}

criterion_group!(benches, bench_ingest);
criterion_main!(benches);
