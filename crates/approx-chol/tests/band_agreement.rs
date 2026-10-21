use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, UpperCsrRef,
};
use approx_chol::{factorize, Factor};

/// `ε^{1/2}`: a column `q e_a` adds exactly `ε` to `a`'s diagonal.
const Q: f64 = 1.0 / (1u64 << 26) as f64;

const MS: std::ops::RangeInclusive<u32> = 1..=40;

fn floats(factor: &Factor) -> bool {
    assert_eq!(factor.blocks().len(), 1, "one block");
    factor.blocks()[0].floats()
}

/// The unit path 0-1-2 plus `m` ε of surplus split over the ends: every entry, `tr = 4 + mε` excepted, is exact.
fn split(m: u32) -> (u32, u32) {
    (m.div_ceil(2), m / 2)
}

fn ends(m: u32) -> (f64, f64) {
    let (a, c) = split(m);
    let eps = f64::EPSILON;
    (1.0 + f64::from(a) * eps, 1.0 + f64::from(c) * eps)
}

fn csr_floats(m: u32) -> bool {
    let (first, last) = ends(m);
    let values = [first, -1.0, -1.0, 2.0, -1.0, -1.0, last];
    let (ptrs, columns) = ([0u32, 2, 5, 7], [0u32, 1, 0, 1, 2, 1, 2]);
    floats(&factorize(CsrRef::new(&ptrs, &columns, &values, 3).unwrap()).unwrap())
}

fn upper_csr_floats(m: u32) -> bool {
    let (first, last) = ends(m);
    let values = [first, -1.0, 2.0, -1.0, last];
    let (ptrs, columns) = ([0u32, 2, 4, 5], [0u32, 1, 1, 2, 2]);
    floats(&factorize(UpperCsrRef::new(&ptrs, &columns, &values, 3).unwrap()).unwrap())
}

fn surplus_columns(m: u32, path: [(u32, u32); 2], ends: [u32; 2], hub: u32) -> Columns {
    let (a, c) = split(m);
    let mut columns = Columns {
        a: path.iter().map(|&(a, _)| a).collect(),
        b: path.iter().map(|&(_, b)| b).collect(),
        x: vec![1.0; 2],
        y: vec![-1.0; 2],
    };
    for (vertex, count) in [(ends[0], a), (ends[1], c)] {
        for _ in 0..count {
            columns.a.push(vertex);
            columns.b.push(hub);
            columns.x.push(Q);
            columns.y.push(0.0);
        }
    }
    columns
}

struct Columns {
    a: Vec<u32>,
    b: Vec<u32>,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn gram_floats(m: u32) -> bool {
    let c = surplus_columns(m, [(0, 1), (1, 2)], [0, 2], 1);
    let gram = FactorWidth2::new(3, (&c.a, &c.x), (&c.b, &c.y)).unwrap();
    floats(&factorize(gram).unwrap())
}

/// The path with its ends `0, 2` on the left as `0, 1` and its middle on the right as `0`.
fn bipartite_floats(m: u32) -> bool {
    let c = surplus_columns(m, [(0, 0), (1, 0)], [0, 1], 0);
    let left = FactorWidth1::new(2, &c.a, Some(&c.x)).unwrap();
    let right = FactorWidth1::new(1, &c.b, Some(&c.y)).unwrap();
    floats(&factorize(BipartiteFactorWidth2::new(left, right).unwrap()).unwrap())
}

fn sddm_floats(m: u32) -> bool {
    let (a, c) = split(m);
    let (u, v) = ([0u32, 1], [1u32, 2]);
    let surplus = [
        f64::from(a) * f64::EPSILON,
        0.0,
        f64::from(c) * f64::EPSILON,
    ];
    let sddm = Sddm::new(Laplacian::new(3, &u, &v).unwrap(), &surplus).unwrap();
    floats(&factorize(sddm).unwrap())
}

/// `1ᵀA1 = mε` against the band `5ε·tr A = 20ε + O(ε²)`.
const EDGE: u32 = 20;

#[test]
fn values_gram_and_its_csr_float_exactly_inside_the_band() {
    let forms = [
        ("csr", csr_floats as fn(u32) -> bool),
        ("upper csr", upper_csr_floats),
        ("gram", gram_floats),
        ("bipartite", bipartite_floats),
    ];
    let wrong: Vec<_> = (forms.iter())
        .flat_map(|&(name, floats)| MS.map(move |m| (name, m, floats(m))))
        .filter(|&(_, m, floats)| floats != (m <= EDGE))
        .collect();
    assert!(wrong.is_empty(), "(form, m, floats): {wrong:?}");
}

/// A stated surplus carries no rounding, so an `Sddm` grounds on it even inside the band.
#[test]
fn sddm_grounds_on_an_in_band_surplus() {
    let floating: Vec<u32> = MS.filter(|&m| sddm_floats(m)).collect();
    assert!(floating.is_empty(), "sddm floats at m = {floating:?}");
}

/// A 9-cycle of `2^1020` weights plus `2^970` of surplus on vertex 0: `tr A` overflows `f64` though every entry fits.
#[test]
fn csr_and_gram_float_alike_where_the_trace_overflows() {
    let (n, w, extra) = (9u32, 2f64.powi(1020), 2f64.powi(970));
    let (mut ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for i in 0..n {
        let mut row = [(i, 2.0 * w), ((i + 1) % n, -w), ((i + n - 1) % n, -w)];
        row.sort_by_key(|&(j, _)| j);
        for (j, x) in row {
            columns.push(j);
            values.push(if i == 0 && j == 0 { x + extra } else { x });
        }
        ptrs.push(columns.len() as u32);
    }
    let csr = floats(&factorize(CsrRef::new(&ptrs, &columns, &values, n).unwrap()).unwrap());
    let (mut a, mut b): (Vec<u32>, Vec<u32>) = (0..n).map(|i| (i, (i + 1) % n)).unzip();
    let root = w.sqrt();
    let (mut x, mut y) = (vec![root; n as usize], vec![-root; n as usize]);
    a.push(0);
    b.push(1);
    x.push(extra.sqrt());
    y.push(0.0);
    let gram =
        floats(&factorize(FactorWidth2::new(n as usize, (&a, &x), (&b, &y)).unwrap()).unwrap());
    assert_eq!((csr, gram), (true, true), "(csr floats, gram floats)");
}
