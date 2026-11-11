use crate::csr::OwnedCsr;
use crate::disjoint::Disjoint;

/// #281's Grams by component sizes, every block relaxed: many components, one connected, and one large beside many small.
pub fn cases() -> [(&'static str, Vec<usize>); 3] {
    let skewed = [[100_000].as_slice(), &[64; 2000]].concat();
    [
        ("1000x2000", vec![2000; 1000]),
        ("connected_200k", vec![200_000]),
        ("skewed_100k_2000x64", skewed),
    ]
}

/// The Gram and its matrix as CSR arrays, the twin `Sgdd::reduce` reads the other way.
pub fn with_csr(sizes: &[usize]) -> (Disjoint, OwnedCsr) {
    let gram = Disjoint::new(sizes, |_| false);
    let columns = (0..gram.a.len()).map(|k| (gram.a[k], gram.b[k], gram.x[k], gram.y[k]));
    let csr = OwnedCsr::gram(gram.n, columns);
    (gram, csr)
}
