#![no_main]

use approx_chol::{BlockFactor, InnerParallelism};
use libfuzzer_sys::fuzz_target;

// Postcard is how `within` persists its factor, so these bytes really do arrive from another process.
fuzz_target!(|data: &[u8]| {
    let Ok(blocks) = postcard::from_bytes::<Vec<BlockFactor<f64, u32>>>(data) else {
        return;
    };
    for block in &blocks {
        let b: Vec<f64> = (0..block.vertices().len())
            .map(|i| (i % 7) as f64 - 3.0)
            .collect();
        let solve = |fill: f64| {
            let mut x = b.clone();
            let mut scratch = vec![fill; block.scratch_len()];
            block
                .solve_local(&mut x, &mut scratch, InnerParallelism::Allow)
                .expect("buffers of the block's own lengths are accepted");
            x
        };
        // NaN scratch reaches the solution if any slot is read before written.
        let agrees =
            (solve(0.0).iter().zip(solve(f64::NAN))).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(agrees, "dirty scratch changed the solution");
    }
});
