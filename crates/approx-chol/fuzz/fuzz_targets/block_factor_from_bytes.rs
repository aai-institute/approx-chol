#![no_main]

use approx_chol::{BlockFactor, MemStack, Threads};
use libfuzzer_sys::fuzz_target;

#[path = "../cells.rs"]
mod cells;

// Postcard is how `within` persists its factor, so these bytes really do arrive from another process.
fuzz_target!(|data: &[u8]| {
    let Ok(blocks) = postcard::from_bytes::<Vec<BlockFactor<f64>>>(data) else {
        return;
    };
    for block in &blocks {
        let b: Vec<f64> = (0..block.vertices().len())
            .map(|i| (i % 7) as f64 - 3.0)
            .collect();
        let solve = |fill: f64| {
            let mut x = b.clone();
            let mut cells = cells::filled(block.scratch(), fill);
            block
                .solve_local(&mut x, MemStack::new_any(&mut cells), Threads::All)
                .expect("a stack of the block's own scratch is accepted");
            x
        };
        // NaN scratch reaches the solution if any slot is read before written.
        let agrees =
            (solve(0.0).iter().zip(solve(f64::NAN))).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(agrees, "dirty scratch changed the solution");
    }
});
