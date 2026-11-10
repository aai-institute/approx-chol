#![no_main]

use approx_chol::{ApproxInverse, MemStack, Threads};
use libfuzzer_sys::fuzz_target;

#[path = "../cells.rs"]
mod cells;

/// `x`'s bits, every NaN the one canonical NaN: Rust leaves a NaN's sign and payload to the optimizer (#437).
fn canonical(x: f64) -> u64 {
    match x.is_nan() {
        true => f64::NAN.to_bits(),
        false => x.to_bits(),
    }
}

/// Bit for bit, a NaN's sign and payload aside.
fn same(a: &[f64], b: impl IntoIterator<Item = f64>) -> bool {
    (a.iter().zip(b)).all(|(&a, b)| canonical(a) == canonical(b))
}

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
            .apply_with(&mut x, Threads::All, MemStack::new_any(&mut cells))
            .expect("a stack of the factor's own scratch is accepted");
        x
    };
    // NaN scratch reaches the solution if any slot is read before written.
    assert!(
        same(&solve(0.0), solve(f64::NAN)),
        "dirty scratch changed the solution"
    );
    let mut x = b.clone();
    factor.apply(&mut x, Threads::All).expect("b has length n");
    assert!(same(&x, solve(0.0)), "apply and apply_with agree");
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
        let mut alone: Vec<f64> = b.iter().map(|&v| sign * v).collect();
        factor
            .apply(&mut alone, Threads::All)
            .expect("b has length n");
        let column = rows.iter().map(|row| row[j]);
        assert!(
            same(&alone, column),
            "each column of a block solves as it does alone"
        );
    }
});
