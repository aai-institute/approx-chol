use std::hint::black_box;
use std::time::Instant;

struct Shape {
    name: String,
    n: usize,
    edges: Vec<(u32, u32, f64)>,
    surplus: f64,
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn grid(side: usize, surplus: f64) -> Shape {
    let mut e = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let i = (r * side + c) as u32;
            if c + 1 < side {
                e.push((i, i + 1, 1.0));
            }
            if r + 1 < side {
                e.push((i, i + side as u32, 1.0));
            }
        }
    }
    let tag = if surplus > 0.0 { "_grounded" } else { "" };
    Shape {
        name: format!("grid{side}{tag}"),
        n: side * side,
        edges: e,
        surplus,
    }
}

fn random_into(e: &mut Vec<(u32, u32, f64)>, offset: usize, n: usize, degree: usize, seed: u64) {
    let mut s = seed;
    for i in 0..n - 1 {
        e.push(((offset + i) as u32, (offset + i + 1) as u32, 1.0));
    }
    for _ in 0..n * degree / 2 {
        let a = (lcg(&mut s) as usize) % n;
        let b = (lcg(&mut s) as usize) % n;
        if a == b {
            continue;
        }
        let w = 0.5 + (lcg(&mut s) % 1000) as f64 / 1000.0;
        e.push(((offset + a.min(b)) as u32, (offset + a.max(b)) as u32, w));
    }
}

fn random(n: usize, degree: usize) -> Shape {
    let mut e = Vec::new();
    random_into(&mut e, 0, n, degree, 42);
    Shape {
        name: format!("rand_n{n}_deg{degree}"),
        n,
        edges: e,
        surplus: 0.0,
    }
}

fn components(k: usize, n: usize, degree: usize) -> Shape {
    let mut e = Vec::new();
    for c in 0..k {
        random_into(&mut e, c * n, n, degree, 42 + c as u64);
    }
    Shape {
        name: format!("disc_{k}x_rand_n{n}_deg{degree}"),
        n: k * n,
        edges: e,
        surplus: 0.0,
    }
}

fn complete(n: usize) -> Shape {
    let mut s = 7u64;
    let mut e = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            e.push((
                a as u32,
                b as u32,
                0.5 + (lcg(&mut s) % 1000) as f64 / 1000.0,
            ));
        }
    }
    Shape {
        name: format!("complete_n{n}"),
        n,
        edges: e,
        surplus: 0.0,
    }
}

fn paths(k: usize, len: usize) -> Shape {
    let mut e = Vec::new();
    for c in 0..k {
        for i in 0..len - 1 {
            e.push(((c * len + i) as u32, (c * len + i + 1) as u32, 1.0));
        }
    }
    Shape {
        name: format!("disc_{k}x_path{len}"),
        n: k * len,
        edges: e,
        surplus: 0.0,
    }
}

struct Csr {
    rp: Vec<u32>,
    ci: Vec<u32>,
    v: Vec<f64>,
}

fn csr(shape: &Shape) -> Csr {
    let mut edges = shape.edges.clone();
    edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let n = shape.n;
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut diag = vec![shape.surplus; n];
    for (i, j, w) in edges {
        rows[i as usize].push((j, -w));
        rows[j as usize].push((i, -w));
        diag[i as usize] += w;
        diag[j as usize] += w;
    }
    let (mut rp, mut ci, mut v) = (vec![0u32], Vec::new(), Vec::new());
    for (i, mut row) in rows.into_iter().enumerate() {
        row.push((i as u32, diag[i]));
        row.sort_by_key(|e| e.0);
        for (j, x) in row {
            ci.push(j);
            v.push(x);
        }
        rp.push(ci.len() as u32);
    }
    Csr { rp, ci, v }
}

fn matvec(a: &Csr, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
        let mut s = 0.0;
        for k in a.rp[i] as usize..a.rp[i + 1] as usize {
            s += a.v[k] * x[a.ci[k] as usize];
        }
        y[i] = s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn pcg(a: &Csr, b: &[f64], precond: &dyn Fn(&[f64]) -> Vec<f64>) -> usize {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let bn = dot(b, b).sqrt();
    let mut z = precond(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    for it in 1..=1000 {
        matvec(a, &p, &mut ap);
        let alpha = rz / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= 1e-8 * bn {
            return it;
        }
        z = precond(&r);
        let rz2 = dot(&r, &z);
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    usize::MAX
}

macro_rules! main_arm {
    ($ac:ident, $a:expr, $n:expr) => {{
        let m = $ac::CsrRef::new(&$a.rp, &$a.ci, &$a.v, $n).unwrap();
        $ac::factorize(m).unwrap()
    }};
}

macro_rules! new_arm {
    ($ac:ident, $a:expr, $n:expr) => {{
        let m = $ac::CsrRef::new(&$a.rp, &$a.ci, &$a.v, $n).unwrap();
        $ac::factorize($ac::Sddm::try_from(m).unwrap())
    }};
}

const ARMS: [&str; 5] = ["main", "main_ctl", "twin", "new", "plain"];

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

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(10);
    let filter = std::env::args().nth(2);
    let shapes = vec![
        grid(100, 0.0),
        grid(100, 1e-3),
        grid(300, 0.0),
        random(20000, 8),
        random(5000, 28),
        random(2000, 12),
        random(300, 40),
        complete(512),
        complete(20),
        paths(2000, 20),
        components(4, 5000, 8),
    ];
    println!("shape,metric,arm,round,us");
    for shape in &shapes {
        if let Some(f) = &filter {
            if !shape.name.contains(f.as_str()) {
                continue;
            }
        }
        let a = csr(shape);
        let n = shape.n as u32;
        let mut s = 99u64;
        let xt: Vec<f64> = (0..shape.n)
            .map(|_| (lcg(&mut s) % 2000) as f64 / 1000.0 - 1.0)
            .collect();
        let mut b = vec![0.0; shape.n];
        matvec(&a, &xt, &mut b);

        let fm = main_arm!(ac_main, a, n);
        let ft = main_arm!(ac_twin, a, n);
        let fnw = new_arm!(ac_new, a, n);
        let fpl = new_arm!(ac_plain, a, n);
        let mut x2 = vec![0.0; fpl.n()];
        let mut scratch2 = vec![0.0; fpl.scratch_len()];
        let (sm, snw) = (fm.solve(&b).unwrap(), fnw.solve(&b).unwrap());
        let diff = sm
            .iter()
            .zip(&snw)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        let it_m = pcg(&a, &b, &|r| fm.solve(r).unwrap());
        let it_t = pcg(&a, &b, &|r| ft.solve(r).unwrap());
        let it_n = pcg(&a, &b, &|r| fnw.solve(r).unwrap());
        eprintln!(
            "{:28} steps main={} new={} | solve max|main-new|={:.2e} identical={} | pcg iters main={} twin={} new={}",
            shape.name, fm.n_steps(), fnw.n_steps(), diff, sm == snw, it_m, it_t, it_n
        );

        let build_reps = reps_for(0.08, || {
            black_box(main_arm!(ac_main, a, n));
        });
        let mut work_m = vec![0.0; fm.n()];
        let mut work_t = vec![0.0; ft.n()];
        let mut x = vec![0.0; fnw.n()];
        let mut scratch = vec![0.0; fnw.scratch_len()];
        let solve_reps = reps_for(0.04, || {
            fm.solve_into(&b, &mut work_m).unwrap();
        });

        for round in 0..rounds {
            let mut order = [0usize, 1, 2, 3, 4];
            order.rotate_left(round % 5);
            if round % 2 == 1 {
                order.reverse();
            }
            for &arm in &order {
                let us = match arm {
                    0 | 1 => time_reps(build_reps, || {
                        black_box(main_arm!(ac_main, a, n));
                    }),
                    2 => time_reps(build_reps, || {
                        black_box(main_arm!(ac_twin, a, n));
                    }),
                    3 => time_reps(build_reps, || {
                        black_box(new_arm!(ac_new, a, n));
                    }),
                    _ => time_reps(build_reps, || {
                        black_box(new_arm!(ac_plain, a, n));
                    }),
                };
                println!("{},build,{},{},{:.3}", shape.name, ARMS[arm], round, us);
            }
            for &arm in &order {
                let us = match arm {
                    0 | 1 => time_reps(solve_reps, || {
                        fm.solve_into(black_box(&b), &mut work_m).unwrap();
                        black_box(&work_m);
                    }),
                    2 => time_reps(solve_reps, || {
                        ft.solve_into(black_box(&b), &mut work_t).unwrap();
                        black_box(&work_t);
                    }),
                    4 => time_reps(solve_reps, || {
                        x2.copy_from_slice(black_box(&b));
                        fpl.solve_in_place(&mut x2, &mut scratch2).unwrap();
                        black_box(&x2);
                    }),
                    _ => time_reps(solve_reps, || {
                        x.copy_from_slice(black_box(&b));
                        fnw.solve_in_place(&mut x, &mut scratch).unwrap();
                        black_box(&x);
                    }),
                };
                println!("{},solve,{},{},{:.3}", shape.name, ARMS[arm], round, us);
            }
        }
        eprintln!("{} done", shape.name);
    }
}
