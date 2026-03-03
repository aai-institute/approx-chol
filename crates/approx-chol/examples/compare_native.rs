use approx_chol::{Builder, Config, CsrRef};
use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Ac,
    Ac2,
}

impl Variant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ac => "ac",
            Self::Ac2 => "ac2",
        }
    }
}

#[derive(Debug)]
struct Options {
    out: String,
    sizes: Vec<(usize, usize)>,
    seeds: Vec<u64>,
    variants: Vec<Variant>,
    factor_warmup: usize,
    factor_repeats: usize,
    apply_repeats: usize,
    tol: f64,
    max_iters: usize,
    split: u32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            out: "target/native_compare/rust.csv".to_string(),
            sizes: vec![(32, 32), (64, 64), (96, 96)],
            seeds: vec![0, 1, 2, 3, 4],
            variants: vec![Variant::Ac, Variant::Ac2],
            factor_warmup: 1,
            factor_repeats: 3,
            apply_repeats: 10,
            tol: 1e-6,
            max_iters: 2000,
            split: 2,
        }
    }
}

#[derive(Debug)]
struct Metrics {
    factor_ms: f64,
    apply_ms: f64,
    pcg_ms: f64,
    pcg_iters: usize,
    pcg_rel_res: f64,
    precond_rel_res: f64,
    n: usize,
}

fn parse_args() -> Result<Options, Box<dyn Error>> {
    let mut opts = Options::default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        let key = &args[i];
        let next = |idx: usize, key: &str, args: &[String]| -> Result<String, Box<dyn Error>> {
            args.get(idx + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {}", key).into())
        };
        match key.as_str() {
            "--out" => {
                opts.out = next(i, key, &args)?;
                i += 2;
            }
            "--sizes" => {
                opts.sizes = parse_sizes(&next(i, key, &args)?)?;
                i += 2;
            }
            "--seeds" => {
                opts.seeds = parse_u64_list(&next(i, key, &args)?)?;
                i += 2;
            }
            "--variants" => {
                opts.variants = parse_variants(&next(i, key, &args)?)?;
                i += 2;
            }
            "--factor-warmup" => {
                opts.factor_warmup = next(i, key, &args)?.parse()?;
                i += 2;
            }
            "--factor-repeats" => {
                opts.factor_repeats = next(i, key, &args)?.parse()?;
                i += 2;
            }
            "--apply-repeats" => {
                opts.apply_repeats = next(i, key, &args)?.parse()?;
                i += 2;
            }
            "--tol" => {
                opts.tol = next(i, key, &args)?.parse()?;
                i += 2;
            }
            "--max-iters" => {
                opts.max_iters = next(i, key, &args)?.parse()?;
                i += 2;
            }
            "--split" => {
                opts.split = next(i, key, &args)?.parse()?;
                i += 2;
            }
            _ => {
                return Err(format!("unknown argument: {}", key).into());
            }
        }
    }

    if opts.sizes.is_empty() {
        return Err("sizes list cannot be empty".into());
    }
    if opts.seeds.is_empty() {
        return Err("seeds list cannot be empty".into());
    }
    if opts.variants.is_empty() {
        return Err("variants list cannot be empty".into());
    }
    if opts.factor_repeats == 0 {
        return Err("factor_repeats must be >= 1".into());
    }
    if opts.apply_repeats == 0 {
        return Err("apply_repeats must be >= 1".into());
    }

    Ok(opts)
}

fn parse_sizes(s: &str) -> Result<Vec<(usize, usize)>, Box<dyn Error>> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let Some((r, c)) = tok.split_once('x') else {
            return Err(format!("invalid size token '{}', expected RxC", tok).into());
        };
        out.push((r.parse()?, c.parse()?));
    }
    Ok(out)
}

fn parse_u64_list(s: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        out.push(tok.parse()?);
    }
    Ok(out)
}

fn parse_variants(s: &str) -> Result<Vec<Variant>, Box<dyn Error>> {
    let mut out = Vec::new();
    for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok {
            "ac" => out.push(Variant::Ac),
            "ac2" => out.push(Variant::Ac2),
            _ => return Err(format!("unknown variant '{}'", tok).into()),
        }
    }
    Ok(out)
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

fn project_zero_mean(x: &mut [f64]) {
    if x.is_empty() {
        return;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    for xi in x.iter_mut() {
        *xi -= mean;
    }
}

fn csr_matvec(row_ptrs: &[u32], col_idx: &[u32], vals: &[f64], x: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        let mut s = 0.0;
        for p in start..end {
            s += vals[p] * x[col_idx[p] as usize];
        }
        *yi = s;
    }
}

fn make_grid_laplacian(rows: usize, cols: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let n = rows * cols;
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    row_ptrs.push(0u32);

    let idx = |r: usize, c: usize| -> usize { r * cols + c };

    for r in 0..rows {
        for c in 0..cols {
            let u = idx(r, c);
            let mut degree = 0.0;

            if r > 0 {
                col_idx.push(idx(r - 1, c) as u32);
                vals.push(-1.0);
                degree += 1.0;
            }
            if r + 1 < rows {
                col_idx.push(idx(r + 1, c) as u32);
                vals.push(-1.0);
                degree += 1.0;
            }
            if c > 0 {
                col_idx.push(idx(r, c - 1) as u32);
                vals.push(-1.0);
                degree += 1.0;
            }
            if c + 1 < cols {
                col_idx.push(idx(r, c + 1) as u32);
                vals.push(-1.0);
                degree += 1.0;
            }

            col_idx.push(u as u32);
            vals.push(degree);
            row_ptrs.push(col_idx.len() as u32);
        }
    }

    (row_ptrs, col_idx, vals)
}

fn make_rhs(n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n];
    if n >= 2 {
        b[0] = 1.0;
        b[n - 1] = -1.0;
    }
    b
}

fn config_for(variant: Variant, seed: u64, split: u32) -> Config {
    match variant {
        Variant::Ac => Config {
            seed,
            split_merge: None,
        },
        Variant::Ac2 => Config {
            seed,
            split_merge: Some(split),
        },
    }
}

fn pcg_with_preconditioner(
    row_ptrs: &[u32],
    col_idx: &[u32],
    vals: &[f64],
    b: &[f64],
    factor: &approx_chol::Factor<f64>,
    tol: f64,
    max_iters: usize,
) -> (usize, f64) {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    project_zero_mean(&mut r);

    let norm_b = norm2(b).max(1e-30);

    let mut z = vec![0.0; n];
    factor.solve_into(&r, &mut z);
    project_zero_mean(&mut z);

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    let mut ap = vec![0.0; n];

    let mut rel_res = norm2(&r) / norm_b;
    if rel_res <= tol {
        return (0, rel_res);
    }

    for iter in 1..=max_iters {
        csr_matvec(row_ptrs, col_idx, vals, &p, &mut ap);
        project_zero_mean(&mut ap);

        let denom = dot(&p, &ap);
        if denom.abs() <= 1e-30 {
            break;
        }

        let alpha = rz_old / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        project_zero_mean(&mut x);
        project_zero_mean(&mut r);

        rel_res = norm2(&r) / norm_b;
        if rel_res <= tol {
            return (iter, rel_res);
        }

        factor.solve_into(&r, &mut z);
        project_zero_mean(&mut z);

        let rz_new = dot(&r, &z);
        if rz_old.abs() <= 1e-30 {
            break;
        }
        let beta = rz_new / rz_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        project_zero_mean(&mut p);
        rz_old = rz_new;
    }

    (max_iters, rel_res)
}

fn run_case(
    rows: usize,
    cols: usize,
    variant: Variant,
    seed: u64,
    opts: &Options,
) -> Result<Metrics, Box<dyn Error>> {
    let (row_ptrs, col_idx, vals) = make_grid_laplacian(rows, cols);
    let n = rows * cols;
    let rhs = make_rhs(n);

    let csr = CsrRef::new(&row_ptrs, &col_idx, &vals, n as u32)?;

    for w in 0..opts.factor_warmup {
        let cfg = config_for(variant, seed.wrapping_add(100_000 + w as u64), opts.split);
        let _ = Builder::<f64>::new(cfg).build(csr)?;
    }

    let mut factor_times = Vec::with_capacity(opts.factor_repeats);
    for r in 0..opts.factor_repeats {
        let cfg = config_for(variant, seed.wrapping_add(r as u64), opts.split);
        let t0 = Instant::now();
        let _ = Builder::<f64>::new(cfg).build(csr)?;
        factor_times.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let cfg_quality = config_for(variant, seed, opts.split);
    let factor = Builder::<f64>::new(cfg_quality).build(csr)?;

    let mut work = vec![0.0; n];
    let mut apply_times = Vec::with_capacity(opts.apply_repeats);
    for _ in 0..opts.apply_repeats {
        let t0 = Instant::now();
        factor.solve_into(&rhs, &mut work);
        apply_times.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    let mut ax = vec![0.0; n];
    csr_matvec(&row_ptrs, &col_idx, &vals, &work, &mut ax);
    let mut residual = vec![0.0; n];
    for i in 0..n {
        residual[i] = rhs[i] - ax[i];
    }
    let precond_rel_res = norm2(&residual) / norm2(&rhs).max(1e-30);

    let t0 = Instant::now();
    let (pcg_iters, pcg_rel_res) = pcg_with_preconditioner(
        &row_ptrs,
        &col_idx,
        &vals,
        &rhs,
        &factor,
        opts.tol,
        opts.max_iters,
    );
    let pcg_ms = t0.elapsed().as_secs_f64() * 1e3;

    Ok(Metrics {
        factor_ms: median(factor_times),
        apply_ms: median(apply_times),
        pcg_ms,
        pcg_iters,
        pcg_rel_res,
        precond_rel_res,
        n,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let opts = parse_args()?;

    if let Some(parent) = std::path::Path::new(&opts.out).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let out = File::create(&opts.out)?;
    let mut writer = BufWriter::new(out);
    writeln!(
        writer,
        "impl,variant,rows,cols,n,seed,factor_ms,apply_ms,pcg_ms,pcg_iters,pcg_rel_res,precond_rel_res"
    )?;

    for &(rows, cols) in &opts.sizes {
        for &variant in &opts.variants {
            for &seed in &opts.seeds {
                let m = run_case(rows, cols, variant, seed, &opts)?;
                writeln!(
                    writer,
                    "rust,{},{},{},{},{},{:.6},{:.6},{:.6},{},{:.6e},{:.6e}",
                    variant.as_str(),
                    rows,
                    cols,
                    m.n,
                    seed,
                    m.factor_ms,
                    m.apply_ms,
                    m.pcg_ms,
                    m.pcg_iters,
                    m.pcg_rel_res,
                    m.precond_rel_res
                )?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}
