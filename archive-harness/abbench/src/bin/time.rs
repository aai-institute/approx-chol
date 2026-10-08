use abbench::{build, build_new, build_typed, csr, rhs, shapes, upper};
use std::hint::black_box;
use std::time::Instant;

const ARMS: [&str; 6] = ["base", "base_ctl", "twin", "new", "newtwin", "typed"];

fn time_reps(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e6
}

fn reps_for(budget_s: f64, mut f: impl FnMut()) -> usize {
    let t = Instant::now();
    f();
    ((budget_s / t.elapsed().as_secs_f64().max(1e-9)) as usize).clamp(3, 5000)
}

fn same(x: &[f64], y: &[f64]) -> bool {
    x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
}

fn main() {
    let rounds: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(10);
    let filter = std::env::args().nth(2).filter(|f| !f.is_empty());
    println!("shape,arm,round,us");
    for shape in &shapes() {
        if filter.as_ref().is_some_and(|f| !shape.name.contains(f.as_str())) {
            continue;
        }
        let a = csr(shape);
        let u = upper(shape);
        let n = shape.n as u32;
        let b = rhs(&a, shape.n);

        let xb = build!(ac_base, a, n).solve(&b).unwrap();
        let xn = build_new!(ac_new, a, n).solve(&b).unwrap();
        let xt = build_typed!(ac_new, u).solve(&b).unwrap();
        eprintln!(
            "{:28} bit-identical new~base {} typed~new {}",
            shape.name,
            same(&xb, &xn),
            same(&xn, &xt)
        );

        let reps = reps_for(0.08, || {
            black_box(build!(ac_base, a, n));
        });
        for round in 0..rounds {
            let mut order = [0usize, 1, 2, 3, 4, 5];
            order.rotate_left(round % 6);
            if round % 2 == 1 {
                order.reverse();
            }
            for &arm in &order {
                let us = match arm {
                    0 | 1 => time_reps(reps, || {
                        black_box(build!(ac_base, a, n));
                    }),
                    2 => time_reps(reps, || {
                        black_box(build!(ac_twin, a, n));
                    }),
                    3 => time_reps(reps, || {
                        black_box(build_new!(ac_new, a, n));
                    }),
                    4 => time_reps(reps, || {
                        black_box(build_new!(ac_newtwin, a, n));
                    }),
                    _ => time_reps(reps, || {
                        black_box(build_typed!(ac_new, u));
                    }),
                };
                println!("{},{},{},{:.3}", shape.name, ARMS[arm], round, us);
            }
        }
        eprintln!("{} done", shape.name);
    }
}
