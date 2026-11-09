use approx_chol::factorize;
use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, TriangleCsrRef,
};

use crate::common::entry::floats;
use crate::common::gram::{dense_csr, matrix};

/// `ε^{1/2}`: a column `q e_a` adds exactly `ε` to `a`'s diagonal.
const Q: f64 = 1.0 / (1u64 << 26) as f64;

const MS: std::ops::RangeInclusive<u32> = 1..=40;

/// `1ᵀA1 = mε` against the band `5ε·tr A = 20ε + O(ε²)`.
const EDGE: u32 = 20;

/// The unit path 0-1-2 plus `m` ε of surplus split over the ends: every entry, `tr = 4 + mε` excepted, is exact.
fn split(m: u32) -> (u32, u32) {
    (m.div_ceil(2), m / 2)
}

struct Columns {
    a: Vec<u32>,
    b: Vec<u32>,
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Columns {
    fn matrix(&self, n: usize) -> Vec<Vec<f64>> {
        matrix(n, &self.a, &self.b, &self.x, &self.y)
    }
}

fn surplus_columns(m: u32, path: [(u32, u32); 2], surplus_at: [u32; 2], hub: u32) -> Columns {
    let (a, c) = split(m);
    let mut columns = Columns {
        a: path.iter().map(|&(a, _)| a).collect(),
        b: path.iter().map(|&(_, b)| b).collect(),
        x: vec![1.0; 2],
        y: vec![-1.0; 2],
    };
    for (vertex, count) in [(surplus_at[0], a), (surplus_at[1], c)] {
        for _ in 0..count {
            columns.a.push(vertex);
            columns.b.push(hub);
            columns.x.push(Q);
            columns.y.push(0.0);
        }
    }
    columns
}

fn path(m: u32) -> Columns {
    surplus_columns(m, [(0, 1), (1, 2)], [0, 2], 1)
}

fn csr_floats(m: u32) -> Vec<bool> {
    let (ptrs, columns, values) = dense_csr(&path(m).matrix(3));
    floats(&factorize(CsrRef::new(3, &ptrs, &columns, &values).unwrap()).unwrap())
}

fn upper_csr_floats(m: u32) -> Vec<bool> {
    let mut matrix = path(m).matrix(3);
    (matrix.iter_mut().enumerate()).for_each(|(i, row)| row[..i].fill(0.0));
    let (ptrs, columns, values) = dense_csr(&matrix);
    floats(&factorize(TriangleCsrRef::upper(3, &ptrs, &columns, &values).unwrap()).unwrap())
}

fn gram_floats(m: u32) -> Vec<bool> {
    let c = path(m);
    let gram = FactorWidth2::new(3, (&c.a, &c.x), (&c.b, &c.y)).unwrap();
    floats(&factorize(gram).unwrap())
}

/// The path with its ends `0, 2` on the left as `0, 1` and its middle on the right as `0`.
fn bipartite_floats(m: u32) -> Vec<bool> {
    let c = surplus_columns(m, [(0, 0), (1, 0)], [0, 1], 0);
    let left = FactorWidth1::valued(2, &c.a, &c.x).unwrap();
    let right = FactorWidth1::valued(1, &c.b, &c.y).unwrap();
    floats(&factorize(BipartiteFactorWidth2::new(left, right).unwrap()).unwrap())
}

fn sddm_floats(surplus: [f64; 3]) -> Vec<bool> {
    let (u, v) = ([0u32, 1], [1u32, 2]);
    let sddm = Sddm::new(Laplacian::new(3, &u, &v).unwrap(), &surplus).unwrap();
    floats(&factorize(sddm).unwrap())
}

fn split_sddm_floats(m: u32) -> Vec<bool> {
    let (a, c) = split(m);
    sddm_floats([
        f64::from(a) * f64::EPSILON,
        0.0,
        f64::from(c) * f64::EPSILON,
    ])
}

#[test]
fn values_gram_and_its_csr_float_exactly_inside_the_band() {
    let forms = [
        ("csr", csr_floats as fn(u32) -> Vec<bool>),
        ("upper csr", upper_csr_floats),
        ("gram", gram_floats),
        ("bipartite", bipartite_floats),
        ("sddm", split_sddm_floats),
    ];
    let wrong: Vec<_> = (forms.iter())
        .flat_map(|&(name, form)| MS.map(move |m| (name, m, form(m))))
        .filter(|(_, m, floated)| *floated != [*m <= EDGE])
        .collect();
    assert!(wrong.is_empty(), "(form, m, floats): {wrong:?}");
}

/// A stated surplus sums no rounded term, so its band is `16ε·d`, not a residue's `(k + 3)ε·d`: `mε` on end 0 floats to `m = 16`.
#[test]
fn an_sddm_stated_surplus_is_rounding_up_to_its_band() {
    let wrong: Vec<(u32, Vec<bool>)> = (MS
        .map(|m| (m, sddm_floats([f64::from(m) * f64::EPSILON, 0.0, 0.0]))))
    .filter(|(m, floated)| *floated != [*m <= 16])
    .collect();
    assert!(wrong.is_empty(), "(m, floats): {wrong:?}");
}

/// A 9-cycle of `2^1020` weights plus `2^970` of surplus on vertex 0: `tr A` overflows `f64` though every entry fits.
#[test]
fn csr_and_gram_float_alike_where_the_trace_overflows() {
    let n = 9u32;
    let (root, extra) = (2f64.powi(510), 2f64.powi(485));
    let (mut a, mut b): (Vec<u32>, Vec<u32>) = (0..n).map(|i| (i, (i + 1) % n)).unzip();
    let (mut x, mut y) = (vec![root; n as usize], vec![-root; n as usize]);
    a.push(0);
    b.push(1);
    x.push(extra);
    y.push(0.0);
    let (ptrs, columns, values) = dense_csr(&matrix(n as usize, &a, &b, &x, &y));
    let csr =
        floats(&factorize(CsrRef::new(n as usize, &ptrs, &columns, &values).unwrap()).unwrap());
    let gram =
        floats(&factorize(FactorWidth2::new(n as usize, (&a, &x), (&b, &y)).unwrap()).unwrap());
    let mut surplus = vec![0.0; n as usize];
    surplus[0] = extra * extra;
    let sddm = sddm_cycle(n, root * root, &surplus);
    assert_eq!(
        (csr, gram, sddm),
        (vec![true], vec![true], vec![true]),
        "(csr floats, gram floats, sddm floats)"
    );
}

fn sddm_cycle(n: u32, weight: f64, surplus: &[f64]) -> Vec<bool> {
    let (u, v): (Vec<u32>, Vec<u32>) = (0..n).map(|i| (i, (i + 1) % n)).unzip();
    let weights = vec![weight; n as usize];
    let laplacian = Laplacian::weighted(n as usize, &u, &v, &weights).unwrap();
    floats(&factorize(Sddm::new(laplacian, surplus).unwrap()).unwrap())
}

/// `8ε·d` on every row of the cycle: each within its band, together past the singular band, as at `2^-1000` the scale, though `tr A` overflows.
#[test]
fn an_sddm_grounds_alike_where_the_trace_overflows() {
    let at = |scale: i32| sddm_cycle(9, 2f64.powi(1020 + scale), &[2f64.powi(972 + scale); 9]);
    assert_eq!((at(0), at(-1000)), (vec![false], vec![false]));
}
