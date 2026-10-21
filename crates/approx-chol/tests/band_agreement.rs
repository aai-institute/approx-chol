use approx_chol::input::{
    BipartiteFactorWidth2, CsrRef, FactorWidth1, FactorWidth2, Laplacian, Sddm, UpperCsrRef,
};
use approx_chol::{factorize, Factor};

/// `ε^{1/2}`: a column `q e_a` adds exactly `ε` to `a`'s diagonal.
const Q: f64 = 1.0 / (1u64 << 26) as f64;

const MS: std::ops::RangeInclusive<u32> = 1..=40;

/// Whether one input form of the path floats at `m` ε of surplus.
type Floats = fn(u32) -> bool;

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

/// The path's columns, then one `q e_end` column per ε of surplus, its other end `hub` at zero.
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

/// `1ᵀA1 = mε` against the band `5ε·tr A ≈ 20ε`: the same matrix floats in every derived form.
#[test]
fn values_gram_and_its_csr_float_alike_across_the_band() {
    let forms: [(&str, Floats); 3] = [
        ("upper csr", upper_csr_floats),
        ("gram", gram_floats),
        ("bipartite", bipartite_floats),
    ];
    let disagree: Vec<_> = (forms.iter())
        .flat_map(|&(name, other)| MS.map(move |m| (name, m, csr_floats(m), other(m))))
        .filter(|&(_, _, csr, other)| csr != other)
        .collect();
    assert!(
        disagree.is_empty(),
        "(form, m, csr floats, form floats): {disagree:?}"
    );
}

/// A stated surplus carries no rounding, so an `Sddm` grounds on it even inside the band.
#[test]
fn sddm_grounds_on_an_in_band_surplus_its_csr_floats_on() {
    let in_band: Vec<u32> = MS.filter(|&m| csr_floats(m)).collect();
    assert!(!in_band.is_empty(), "some surplus lies in the band");
    let floating: Vec<u32> = MS.filter(|&m| sddm_floats(m)).collect();
    assert!(floating.is_empty(), "sddm floats at m = {floating:?}");
}
