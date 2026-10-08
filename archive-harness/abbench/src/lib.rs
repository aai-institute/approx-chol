pub struct Shape {
    pub name: String,
    pub n: usize,
    pub edges: Vec<(u32, u32, f64)>,
    /// Added to the diagonal of every `every`-th vertex; 0 = none.
    pub surplus: f64,
    pub every: usize,
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn shape(name: String, n: usize, edges: Vec<(u32, u32, f64)>) -> Shape {
    Shape { name, n, edges, surplus: 0.0, every: 1 }
}

fn grounded(mut s: Shape, every: usize) -> Shape {
    s.name = format!("{}_g{every}", s.name);
    s.surplus = 1e-3;
    s.every = every;
    s
}

fn grid(side: usize) -> Shape {
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
    shape(format!("grid{side}"), side * side, e)
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
    shape(format!("rand_n{n}_deg{degree}"), n, e)
}

fn components(k: usize, n: usize, degree: usize) -> Shape {
    let mut e = Vec::new();
    for c in 0..k {
        random_into(&mut e, c * n, n, degree, 42 + c as u64);
    }
    shape(format!("disc_{k}x_rand_n{n}_deg{degree}"), k * n, e)
}

fn complete(n: usize) -> Shape {
    let mut s = 7u64;
    let mut e = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            e.push((a as u32, b as u32, 0.5 + (lcg(&mut s) % 1000) as f64 / 1000.0));
        }
    }
    shape(format!("complete_n{n}"), n, e)
}

fn paths(k: usize, len: usize) -> Shape {
    let mut e = Vec::new();
    for c in 0..k {
        for i in 0..len - 1 {
            e.push(((c * len + i) as u32, (c * len + i + 1) as u32, 1.0));
        }
    }
    shape(format!("disc_{k}x_path{len}"), k * len, e)
}

fn isolated(n: usize) -> Shape {
    shape(format!("isolated_n{n}"), n, Vec::new())
}

pub fn shapes() -> Vec<Shape> {
    vec![
        grid(100),
        grounded(grid(100), 1),
        grid(300),
        random(20000, 8),
        random(5000, 28),
        grounded(random(5000, 28), 10),
        random(2000, 12),
        random(300, 40),
        complete(512),
        complete(20),
        paths(2000, 20),
        grounded(paths(2000, 20), 20),
        paths(50000, 2),
        components(4, 5000, 8),
        grounded(isolated(1_000_000), 1),
    ]
}

pub struct Csr {
    pub rp: Vec<u32>,
    pub ci: Vec<u32>,
    pub v: Vec<f64>,
}

pub fn csr(shape: &Shape) -> Csr {
    let mut edges = shape.edges.clone();
    edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let n = shape.n;
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut diag: Vec<f64> = (0..n)
        .map(|i| if shape.surplus > 0.0 && i % shape.every == 0 { shape.surplus } else { 0.0 })
        .collect();
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

pub fn matvec(a: &Csr, x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            (a.rp[i] as usize..a.rp[i + 1] as usize)
                .map(|k| a.v[k] * x[a.ci[k] as usize])
                .sum()
        })
        .collect()
}

pub fn rhs(a: &Csr, n: usize) -> Vec<f64> {
    let mut s = 99u64;
    let xt: Vec<f64> = (0..n).map(|_| (lcg(&mut s) % 2000) as f64 / 1000.0 - 1.0).collect();
    matvec(a, &xt)
}

/// The strict upper adjacency and, when grounded, each vertex's surplus.
pub struct Upper {
    pub rp: Vec<u32>,
    pub nb: Vec<u32>,
    pub w: Vec<f64>,
    pub surplus: Option<Vec<f64>>,
}

pub fn upper(shape: &Shape) -> Upper {
    let mut edges = shape.edges.clone();
    edges.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    edges.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    let mut rp = vec![0u32];
    let (mut nb, mut w) = (Vec::new(), Vec::new());
    let mut k = 0;
    for row in 0..shape.n as u32 {
        while k < edges.len() && edges[k].0 == row {
            nb.push(edges[k].1);
            w.push(edges[k].2);
            k += 1;
        }
        rp.push(nb.len() as u32);
    }
    let surplus = (shape.surplus > 0.0).then(|| {
        (0..shape.n)
            .map(|i| if i % shape.every == 0 { shape.surplus } else { 0.0 })
            .collect()
    });
    Upper { rp, nb, w, surplus }
}

#[macro_export]
macro_rules! build_new {
    ($ac:ident, $a:expr, $n:expr) => {{
        let m = $ac::CsrRef::new(&$a.rp, &$a.ci, &$a.v, $n).unwrap();
        $ac::factorize($ac::Sddm::try_from(m).unwrap())
    }};
}

#[macro_export]
macro_rules! build_typed {
    ($ac:ident, $u:expr) => {{
        let u = &$u;
        let sddm = match &u.surplus {
            None => $ac::Sddm::from(
                $ac::Laplacian::new(u.rp.clone(), u.nb.clone(), u.w.clone()).unwrap(),
            ),
            Some(s) => $ac::Sddm::new(u.rp.clone(), u.nb.clone(), u.w.clone(), s.clone()).unwrap(),
        };
        $ac::factorize(sddm)
    }};
}

#[macro_export]
macro_rules! build {
    ($ac:ident, $a:expr, $n:expr) => {{
        let m = $ac::CsrRef::new(&$a.rp, &$a.ci, &$a.v, $n).unwrap();
        $ac::factorize(m).unwrap()
    }};
}
