#![no_main]

use approx_chol::Factor;
use libfuzzer_sys::fuzz_target;

// A persisted factor's blocks must cover `0..n` once each, or a solve would index past `x`.
fuzz_target!(|data: &[u8]| {
    let Ok(factor) = postcard::from_bytes::<Factor<f64>>(data) else {
        return;
    };
    let b: Vec<f64> = (0..factor.n()).map(|i| (i % 7) as f64 - 3.0).collect();
    let solve = |fill: f64| {
        let mut x = b.clone();
        let mut scratch = vec![fill; factor.scratch_len()];
        factor
            .solve_in_place(&mut x, &mut scratch)
            .expect("buffers of the factor's own lengths are accepted");
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
});
