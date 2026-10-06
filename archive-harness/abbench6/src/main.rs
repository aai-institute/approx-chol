struct Shape {
    name: String,
    n: usize,
    upper: Vec<Vec<(u32, f64)>>,
    surplus: Vec<f64>,
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn from_edges(name: &str, n: usize, mut edges: Vec<(u32, u32, f64)>, surplus: Vec<f64>) -> Shape {
    edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let mut upper = vec![Vec::new(); n];
    for (u, v, w) in edges {
        upper[u as usize].push((v, w));
    }
    Shape {
        name: name.to_owned(),
        n,
        upper,
        surplus,
    }
}

fn grid_edges(side: usize) -> Vec<(u32, u32, f64)> {
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
    e
}

fn random_edges(n: usize, degree: usize, seed: u64) -> Vec<(u32, u32, f64)> {
    let mut s = seed;
    let mut e = Vec::new();
    for i in 0..n - 1 {
        e.push((i as u32, i as u32 + 1, 1.0));
    }
    for _ in 0..n * degree / 2 {
        let a = (lcg(&mut s) as usize) % n;
        let b = (lcg(&mut s) as usize) % n;
        if a == b {
            continue;
        }
        let w = 0.5 + (lcg(&mut s) % 1000) as f64 / 1000.0;
        e.push((a.min(b) as u32, a.max(b) as u32, w));
    }
    e
}

fn complete_edges(n: usize) -> Vec<(u32, u32, f64)> {
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
    e
}

fn paths_edges(parts: usize, len: usize) -> Vec<(u32, u32, f64)> {
    let mut e = Vec::new();
    for p in 0..parts {
        for i in 0..len - 1 {
            let v = (p * len + i) as u32;
            e.push((v, v + 1, 1.0));
        }
    }
    e
}

fn every(n: usize, stride: usize, value: f64) -> Vec<f64> {
    (0..n)
        .map(|v| if v % stride == 0 { value } else { 0.0 })
        .collect()
}

fn few(n: usize, count: usize, value: f64) -> Vec<f64> {
    let mut s = vec![0.0; n];
    for k in 0..count {
        s[(k * 7919 + 13) % n] = value;
    }
    s
}

fn upper_arrays(shape: &Shape) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut nb, mut w) = (vec![0u32], Vec::new(), Vec::new());
    for row in &shape.upper {
        for &(j, x) in row {
            nb.push(j);
            w.push(x);
        }
        rp.push(nb.len() as u32);
    }
    (rp, nb, w)
}

fn grounded(shape: &Shape) -> bool {
    shape.surplus.iter().any(|&s| s > 0.0)
}

macro_rules! sddm {
    ($ac:ident, $shape:expr) => {{
        let (rp, nb, w) = upper_arrays($shape);
        let l = $ac::Laplacian::new(rp, nb, w).unwrap();
        let input: $ac::Sddm = if grounded($shape) {
            $ac::Grounded::new(l, $shape.surplus.clone())
                .unwrap()
                .into()
        } else {
            l.into()
        };
        input
    }};
}

macro_rules! build {
    ($ac:ident, $sddm:expr, $seed:expr) => {{
        $ac::factorize_with(
            $sddm,
            $ac::Config {
                seed: $seed,
                ..$ac::Config::default()
            },
        )
        .unwrap()
    }};
}

fn matvec(shape: &Shape, x: &[f64], y: &mut [f64]) {
    for (i, (yi, &s)) in y.iter_mut().zip(&shape.surplus).enumerate() {
        *yi = s * x[i];
    }
    for (i, row) in shape.upper.iter().enumerate() {
        for &(j, w) in row {
            let d = w * (x[i] - x[j as usize]);
            y[i] += d;
            y[j as usize] -= d;
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn pcg(shape: &Shape, b: &[f64], precond: &mut dyn FnMut(&mut [f64])) -> usize {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let bn = dot(b, b).sqrt();
    let mut z = r.clone();
    precond(&mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    for it in 1..=1000 {
        matvec(shape, &p, &mut ap);
        let alpha = rz / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() / bn < 1e-8 {
            return it;
        }
        z.copy_from_slice(&r);
        precond(&mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    1000
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn rhs(shape: &Shape) -> Vec<f64> {
    let mut s = 99u64;
    let mut b: Vec<f64> = (0..shape.n)
        .map(|_| (lcg(&mut s) % 2_000_001) as f64 / 1e6 - 1.0)
        .collect();
    if !grounded(shape) {
        let mean = b.iter().sum::<f64>() / b.len() as f64;
        for v in &mut b {
            *v -= mean;
        }
    }
    b
}

fn shapes() -> Vec<Shape> {
    let side = 300;
    let gn = side * side;
    vec![
        from_edges("F_grid300", gn, grid_edges(side), vec![0.0; gn]),
        from_edges("F_grid100", 10_000, grid_edges(100), vec![0.0; 10_000]),
        from_edges(
            "F_rand100k_deg8",
            100_000,
            random_edges(100_000, 8, 42),
            vec![0.0; 100_000],
        ),
        from_edges("G_grid300_one", gn, grid_edges(side), few(gn, 1, 1.0)),
        from_edges("G_grid300_few4", gn, grid_edges(side), few(gn, 4, 1.0)),
        from_edges("G_grid300_all", gn, grid_edges(side), vec![1e-3; gn]),
        from_edges(
            "G_rand5000_deg28_few4",
            5000,
            random_edges(5000, 28, 5),
            few(5000, 4, 1.0),
        ),
        from_edges(
            "G_complete400_all",
            400,
            complete_edges(400),
            vec![0.05; 400],
        ),
    ]
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    if mode == "quality" {
        println!("shape,arm,seed,value");
        for shape in &shapes() {
            let b = rhs(shape);
            for seed in 0..10u64 {
                let fb = build!(ac_base, sddm!(ac_base, shape), seed);
                let ff = build!(ac_forced, sddm!(ac_forced, shape), seed);
                let fnw = build!(ac_new, sddm!(ac_new, shape), seed);
                let mut sb = vec![0.0; fb.scratch_len()];
                let mut sf = vec![0.0; ff.scratch_len()];
                let mut sn = vec![0.0; fnw.scratch_len()];
                println!(
                    "{},base,{seed},{}",
                    shape.name,
                    pcg(shape, &b, &mut |z| fb.solve_in_place(z, &mut sb).unwrap())
                );
                println!(
                    "{},forced,{seed},{}",
                    shape.name,
                    pcg(shape, &b, &mut |z| ff.solve_in_place(z, &mut sf).unwrap())
                );
                println!(
                    "{},new,{seed},{}",
                    shape.name,
                    pcg(shape, &b, &mut |z| fnw.solve_in_place(z, &mut sn).unwrap())
                );
            }
        }
        return;
    }
    println!("shape,arm,round,value");
    for shape in &shapes() {
        let reps = if shape.n > 50_000 { 5 } else { 15 };
        let b = rhs(shape);
        let mut sink = 0usize;
        for round in 0..8 {
            let arms = ["base", "base_ctl", "new"];
            let order: Vec<usize> = if round % 2 == 0 {
                vec![0, 1, 2]
            } else {
                vec![2, 1, 0]
            };
            for &a in &order {
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    if a == 2 {
                        let s = sddm!(ac_new, shape);
                        let t = std::time::Instant::now();
                        let f = build!(ac_new, s, 0);
                        times.push(t.elapsed().as_secs_f64());
                        sink += f.n_steps();
                    } else {
                        let s = sddm!(ac_base, shape);
                        let t = std::time::Instant::now();
                        let f = build!(ac_base, s, 0);
                        times.push(t.elapsed().as_secs_f64());
                        sink += f.n_steps();
                    }
                }
                println!(
                    "{},build_{},{round},{:.1}",
                    shape.name,
                    arms[a],
                    median(times) * 1e6
                );
            }
        }
        let fb = build!(ac_base, sddm!(ac_base, shape), 0);
        let fnw = build!(ac_new, sddm!(ac_new, shape), 0);
        let mut sb = vec![0.0; fb.scratch_len()];
        let mut sn = vec![0.0; fnw.scratch_len()];
        let mut x = vec![0.0; shape.n];
        for round in 0..8 {
            let order: Vec<usize> = if round % 2 == 0 {
                vec![0, 1, 2]
            } else {
                vec![2, 1, 0]
            };
            for &a in &order {
                let mut times = Vec::with_capacity(21);
                for _ in 0..21 {
                    x.copy_from_slice(&b);
                    let t = std::time::Instant::now();
                    if a == 2 {
                        fnw.solve_in_place(&mut x, &mut sn).unwrap()
                    } else {
                        fb.solve_in_place(&mut x, &mut sb).unwrap()
                    }
                    times.push(t.elapsed().as_secs_f64());
                }
                println!(
                    "{},solve_{},{round},{:.1}",
                    shape.name,
                    ["base", "base_ctl", "new"][a],
                    median(times) * 1e6
                );
            }
        }
        eprintln!("{} done ({sink})", shape.name);
    }
}
