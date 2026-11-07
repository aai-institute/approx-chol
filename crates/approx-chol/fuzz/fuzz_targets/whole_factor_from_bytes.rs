#![no_main]

use approx_chol::{ApproxInverse, MemStack, Threads};
use libfuzzer_sys::fuzz_target;

#[path = "../cells.rs"]
mod cells;

// A persisted factor's blocks must cover `0..n` once each, or a solve would index past `x`.
fuzz_target!(|data: &[u8]| {
    let Ok(factor) = postcard::from_bytes::<ApproxInverse<f64>>(data) else {
        return;
    };
    let b: Vec<f64> = (0..factor.n()).map(|i| (i % 7) as f64 - 3.0).collect();
    let solve = |fill: f64| {
        let mut x = b.clone();
        let mut cells = cells::filled(factor.scratch(1, Threads::All), fill);
        factor
            .solve_in_place(&mut x, MemStack::new_any(&mut cells))
            .expect("a stack of the factor's own scratch is accepted");
        x
    };
    // NaN scratch reaches the solution if any slot is read before written.
    let agrees = (solve(0.0).iter().zip(solve(f64::NAN))).all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(agrees, "dirty scratch changed the solution");
    let x = factor.solve(&b).expect("b has length n");
    assert!(
        x.iter()
            .zip(solve(0.0))
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "solve and solve_in_place agree"
    );
    let mut rows: Vec<[f64; 3]> = b.iter().map(|&v| [v, -v, 0.5 * v]).collect();
    let mut cells = cells::filled(factor.scratch(3, Threads::All), f64::NAN);
    factor
        .apply_with(
            rows.as_mut_slice(),
            Threads::All,
            MemStack::new_any(&mut cells),
        )
        .expect("a stack of the factor's own scratch is accepted");
    for (j, sign) in [1.0, -1.0, 0.5].into_iter().enumerate() {
        let alone = factor
            .solve(&b.iter().map(|&v| sign * v).collect::<Vec<_>>())
            .expect("b has length n");
        let column = rows.iter().map(|row| row[j]);
        assert!(
            column.zip(alone).all(|(a, b)| a.to_bits() == b.to_bits()),
            "each column of a block solves as it does alone"
        );
    }
});
