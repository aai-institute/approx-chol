//! Allocation totals and peak bytes per shape. Peak is the number a flat edge
//! buffer trades against: it removes one allocation per row but keeps dead space
//! a released `Vec` would have returned.

mod common;

use approx_chol::low_level::Builder;
use approx_chol::Config;
use common::shapes::sweep;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    for (label, lap) in sweep() {
        let csr = lap.as_csr_ref();
        let builder = Builder::<f64>::new(Config::default());
        let profiler = dhat::Profiler::builder().testing().build();
        let factor = builder.build(csr).expect("sweep shapes must factorize");
        let stats = dhat::HeapStats::get();
        drop(profiler);
        let rows = csr.n();
        println!(
            "{label}: n={rows} allocs={} ({:.2}/row) peak={} bytes total={} bytes",
            stats.total_blocks,
            stats.total_blocks as f64 / rows as f64,
            stats.max_bytes,
            stats.total_bytes,
        );
        drop(factor);
    }
}
