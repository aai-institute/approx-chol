//! A row's surplus within its band is rounding, never a ground (#311, #343).
use std::collections::BTreeMap;

use approx_chol::input::{CsrRef, FactorWidth2};
use approx_chol::{factorize, ApproxInverse, Config, Real};

use crate::common::bits::bits;
use crate::common::entry::{by_type, floats, Entry, Factored, Types};
use crate::common::graph::{Edge, Graph};
use crate::common::pcg::pcg_iterations;

/// xoshiro256** seeded through SplitMix64, as #325 and #343 drew their inputs.
struct Rng([u64; 4]);

impl Rng {
    fn new(mut z: u64) -> Self {
        let mut next = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let x = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        (s[2], s[3]) = (s[2] ^ s[0], s[3] ^ s[1]);
        (s[1], s[0]) = (s[1] ^ s[2], s[0] ^ s[3]);
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }
}

/// #325's generator: a tree over a random permutation, random pairs to `n·degree/2` edges, repeats summed, and `n/every` rows grounded by `ground`.
fn graph(
    n: usize,
    degree: usize,
    (ground, every): (f64, usize),
    seed: u64,
    weight: impl Fn(&mut Rng) -> f64,
) -> Graph {
    let mut rng = Rng::new(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        perm.swap(i, j);
    }
    let mut edges = BTreeMap::new();
    let mut add =
        |a: usize, b: usize, w: f64| *edges.entry((a.min(b), a.max(b))).or_insert(0.0) += w;
    for i in 1..n {
        let (j, w) = (rng.below(i), weight(&mut rng));
        add(perm[i], perm[j], w);
    }
    let mut placed = n - 1;
    while placed < n * degree / 2 {
        let (a, b) = (rng.below(n), rng.below(n));
        if a != b {
            let w = weight(&mut rng);
            add(a, b, w);
            placed += 1;
        }
    }
    let edges = edges
        .into_iter()
        .map(|((a, b), w)| Edge::negative(a, b, w))
        .collect();
    let mut graph = Graph::new(n, edges);
    let mut rows: Vec<usize> = (0..n).collect();
    for i in 0..n / every {
        let j = i + rng.below(n - i);
        rows.swap(i, j);
        graph.ground[rows[i]] = ground;
    }
    graph
}

fn factored(entry: Entry, graph: &Graph, types: Types) -> Factored {
    entry.reduce(graph, types).factorize(&Config::default())
}

fn floats_of(factored: &Factored) -> Vec<bool> {
    by_type!(&factored.factor, factor => floats(factor))
}

/// #343's inputs: grounds of `1` beside weights to `1e14` are below `f32`'s resolution, so the `Sddm` floats as its CSR does, bit for bit, and preconditions a consistent system to convergence.
#[test]
fn an_f32_sddm_with_sub_resolution_grounds_floats_as_its_csr() {
    for seed in 1..=3 {
        let graph = graph(2000, 30, (1.0, 1000), seed, |rng| {
            10f64.powf(14.0 * rng.unit())
        });
        let sddm = factored(Entry::Sddm, &graph, Types::F32U32);
        let csr = factored(Entry::Csr, &graph, Types::F32U32);
        assert_eq!(
            (floats_of(&sddm), floats_of(&csr)),
            (vec![true], vec![true]),
            "seed {seed}"
        );
        let x: Vec<f64> = (0..graph.n).map(|i| [1.0, -1.0, 0.5, 2.0][i % 4]).collect();
        let b = graph.apply(&x);
        assert_eq!(bits(&sddm.solve(&b)), bits(&csr.solve(&b)), "seed {seed}");
        let iterations = pcg_iterations(|x| graph.apply(x), |r| sddm.solve(r), &b, 1e-8, 200);
        assert!(iterations.is_some(), "seed {seed}");
    }
}

fn f32_csr((ptrs, columns, values): &(Vec<usize>, Vec<usize>, Vec<f64>)) -> ApproxInverse<f32> {
    let ptrs: Vec<u32> = ptrs.iter().map(|&p| p as u32).collect();
    let columns: Vec<u32> = columns.iter().map(|&j| j as u32).collect();
    let values: Vec<f32> = values.iter().map(|&a| a as f32).collect();
    factorize(CsrRef::new(ptrs.len() - 1, &ptrs, &columns, &values).unwrap()).unwrap()
}

/// #311: a grounded `f32` CSR at weight span `1e16` factors as the same CSR whose ungrounded rows hold no residue.
#[test]
fn a_grounded_f32_csr_keeps_no_rounding_residue_as_a_ground() {
    let graph = graph(2000, 10, (1.0, 100), 311, |rng| {
        10f64.powf(16.0 * (rng.unit() - 0.5))
    });
    let rounded = graph.csr(false);
    let (ptrs, columns, mut values) = rounded.clone();
    for i in (0..graph.n).filter(|&i| graph.ground[i] == 0.0) {
        let row = ptrs[i]..ptrs[i + 1];
        let off = (row.clone()).filter(|&k| columns[k] != i);
        let sum: f64 = off.map(|k| f64::from((values[k] as f32).abs())).sum();
        (row.filter(|&k| columns[k] == i)).for_each(|k| values[k] = f64::from(sum as f32));
    }
    let (factor, exact) = (f32_csr(&rounded), f32_csr(&(ptrs, columns, values)));
    assert_eq!(floats(&factor), [false]);
    let b: Vec<f32> = (0..graph.n).map(|i| [1.0, -1.0, 0.5, 2.0][i % 4]).collect();
    let solve = |f: &ApproxInverse<f32>| crate::common::solve::Solve::solve(f, &b).unwrap();
    assert_eq!(solve(&factor), solve(&exact));
}

/// `K_31` with unit weights, each `d = 30`, and `ulps · 30ε` more on vertex 0: a stated ground passes its band, `16ε·d`, from 17, a residue its `(k + 3)ε·d` from 34.
#[test]
fn a_stated_ground_past_its_band_grounds_where_the_same_residue_is_rounding() {
    let edges = (0..31).flat_map(|a| (a + 1..31).map(move |b| Edge::negative(a, b, 1.0)));
    let complete = Graph::new(31, edges.collect());
    let floats = |entry: Entry, ulps: f64| {
        let mut graph = complete.clone();
        graph.ground[0] = ulps * 30.0 * f64::EPSILON;
        floats_of(&factored(entry, &graph, Types::F64U32))
    };
    assert_eq!(
        (floats(Entry::Sddm, 15.0), floats(Entry::Sddm, 27.0)),
        (vec![true], vec![false])
    );
    assert_eq!(floats(Entry::Csr, 27.0), [true]);
}

/// A path of columns `(1.3, −2.7)`, `(2.7, −1.3)` alternating, not dominant unscaled, so scaled by its null vector, plus a one-ended column stating `ratio · a_44` on vertex 4, `a_44 = 2·1.3²` before it: its band is the null vector's of no summed term, `3ε + 16ε`.
fn scaled_gram_floats(ratio: f64) -> Vec<bool> {
    let n = 9u32;
    let (mut a, mut b): (Vec<u32>, Vec<u32>) = (0..n - 1).map(|i| (i, i + 1)).unzip();
    let (mut x, mut y): (Vec<f64>, Vec<f64>) = (0..n - 1)
        .map(|i| if i % 2 == 0 { (1.3, -2.7) } else { (2.7, -1.3) })
        .unzip();
    a.push(4);
    b.push(0);
    x.push((ratio / (1.0 - ratio) * 2.0 * 1.3f64.powi(2)).sqrt());
    y.push(0.0);
    floats(&factorize(FactorWidth2::new(n as usize, (&a, &x), (&b, &y)).unwrap()).unwrap())
}

#[test]
fn a_scaled_grams_stated_ground_is_rounding_within_its_scalings_band() {
    let epsilon = f64::EPSILON;
    assert_eq!(scaled_gram_floats(0.0), [true]);
    assert_eq!(scaled_gram_floats(18.0 * epsilon), [true]);
    assert_eq!(scaled_gram_floats(20.0 * epsilon), [false]);
}

/// A star of 24 leaves short of dominance by `20ε` each, its leaves joined pairwise, and a stated ground of `17ε·d_0` on its centre: past its band, but on a row whose residue is a deficit, so not a ground; the block grounds on its residues.
fn star_factors<T: Real>() -> bool {
    let (k, epsilon) = (24u32, T::epsilon());
    let leg = T::one() + T::from(20.0).unwrap() * epsilon;
    let mut columns: Vec<(u32, T, u32, T)> = Vec::new();
    for l in 1..=k {
        columns.push((0, T::one(), l, -leg));
        columns.extend((l + 1..=k).map(|m| (l, T::one(), m, -T::one())));
    }
    let z = (17.0 * epsilon.to_f64().unwrap() * f64::from(k) * 1.0001).sqrt();
    columns.push((0, T::from(z).unwrap(), 1, T::zero()));
    let a: Vec<u32> = columns.iter().map(|c| c.0).collect();
    let x: Vec<T> = columns.iter().map(|c| c.1).collect();
    let b: Vec<u32> = columns.iter().map(|c| c.2).collect();
    let y: Vec<T> = columns.iter().map(|c| c.3).collect();
    factorize(FactorWidth2::new(k as usize + 1, (&a, &x), (&b, &y)).unwrap()).is_ok()
}

#[test]
fn a_stated_ground_on_a_deficit_row_is_no_ground() {
    assert!(star_factors::<f32>());
    assert!(star_factors::<f64>());
}
