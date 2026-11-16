use crate::common::cases::cases;
use crate::common::entry::{Entry, Types};
use crate::common::graph::{Edge, Graph};
use crate::common::modes::Mode;
use crate::common::solve::Solve;
use crate::common::sparse::Sparse;

const RHS: [f64; 6] = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5];

/// Uniform conductance `w`, so scaling `w` scales the whole matrix and nothing else.
fn complete(w: f64) -> Graph {
    let pairs = (0..6).flat_map(|a| (a + 1..6).map(move |b| (a, b)));
    Graph::new(6, pairs.map(|(a, b)| Edge::negative(a, b, w)).collect())
}

/// As [`complete`] on two sides of three, which a bipartite Gram states too.
fn complete_bipartite(w: f64) -> Graph {
    let pairs = (0..3).flat_map(|a| (3..6).map(move |b| (a, b)));
    Graph::new(6, pairs.map(|(a, b)| Edge::negative(a, b, w)).collect())
}

/// As [`complete`] with every vertex tied to ground by `w`, so no `x_i` is zero.
fn grounded_complete(w: f64) -> Graph {
    let mut graph = complete(w);
    graph.ground.fill(w);
    graph
}

/// Binary exponents of `w`: a power of ten rounds each weight apart, and a fill sum tied at `w = 1` then breaks its tie by summation order (`2^-53` measured 1.2%).
const F64_EXPONENTS: [i32; 20] = [
    -963, -664, -332, -100, -53, -50, -47, -17, -3, 70, 76, 103, 106, 123, 143, 166, 332, 505, 664,
    997,
];
const F32_EXPONENTS: [i32; 14] = [
    -100, -66, -40, -27, -23, -20, -17, -7, -3, 23, 33, 40, 66, 100,
];

/// `w · x(w)` is invariant at a fixed seed, so an absolute floor deviates by order one.
fn assert_invariant_under_scaling(
    entry: Entry,
    fixture: fn(f64) -> Graph,
    types: Types,
    mode: Mode,
) {
    // Measured at most 290ε, flat across exponents.
    let tolerance = 1024.0 * types.epsilon();
    let config = mode.config();
    let solve = |w: f64| {
        let reduced = entry.reduce(&fixture(w), types);
        let x = reduced.factorize(&config).solve(&RHS);
        x.iter().map(|&x| x * w).collect::<Vec<f64>>()
    };
    let exponents = match types {
        Types::F32U32 | Types::F32U64 => &F32_EXPONENTS[..],
        Types::F64U32 | Types::F64U64 => &F64_EXPONENTS[..],
    };
    let want = solve(1.0);
    for &exponent in exponents {
        let scaled = solve(2f64.powi(exponent));
        for (index, (&scaled, &want)) in scaled.iter().zip(&want).enumerate() {
            let deviation = ((scaled - want) / want).abs();
            assert!(
                deviation < tolerance,
                "{types:?} {config:?} w=2^{exponent}: x[{index}] deviates by {deviation:e}"
            );
        }
    }
}

cases! {
    fn factorization_is_invariant_under_uniform_scaling(
        route in routes(floating_non_bipartite),
        types in types(all),
    ) {
        let (entry, mode) = route;
        assert_invariant_under_scaling(entry, complete, types, mode);
    }
}

/// The typed route only: CSR's approximate route deviates 3–5% at `w = 1e-200`.
#[rstest::rstest]
fn an_sddm_factorization_is_invariant_under_uniform_scaling(
    #[values(Types::F32U32, Types::F32U64, Types::F64U32, Types::F64U64)] types: Types,
    #[values(
        Mode::Dense,
        Mode::Sparse(Sparse::AcExactComplement),
        Mode::Sparse(Sparse::Ac2ExactComplement)
    )]
    mode: Mode,
) {
    assert_invariant_under_scaling(Entry::Sddm, grounded_complete, types, mode);
}

cases! {
    fn bipartite_factorization_is_invariant_under_uniform_scaling(
        route in routes(floating),
        types in types(all),
    ) {
        let (entry, mode) = route;
        assert_invariant_under_scaling(entry, complete_bipartite, types, mode);
    }
}

/// A bipartite path whose eliminated row joins `big²` to `floor²` at the range floor: `Σ c_ek x_k` overflowed before `D_e⁻¹` scaled it.
fn path_solves_as_its_scaled_rhs<T: approx_chol::Real>(big: T, floor: T, down: i32) {
    use approx_chol::input::{BipartiteFactorWidth2, FactorWidth1};
    let s = [big, floor, floor, floor];
    let solve = |b: &[T]| {
        let gram = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(3, &[0u32, 1, 1, 2]).unwrap(),
            FactorWidth1::new(2, &[0u32, 0, 1, 1]).unwrap(),
            &s,
        )
        .unwrap();
        approx_chol::factorize(gram).unwrap().solve(b).unwrap()
    };
    let b: Vec<T> = [1.0, -1.0, 0.0, 0.0, 0.0]
        .map(|v| T::from(v).unwrap())
        .to_vec();
    let scale = T::from(2f64.powi(down)).unwrap();
    let x = solve(&b);
    let scaled = solve(&b.iter().map(|&v| v * scale).collect::<Vec<_>>());
    let peak = x.iter().fold(T::zero(), |m, v| m.max(v.abs()));
    assert!(peak.is_finite(), "{x:?}");
    for (x, s) in x.iter().zip(&scaled) {
        assert!((*x * scale - *s).abs() <= T::epsilon() * peak * scale * T::from(16).unwrap());
    }
}

#[test]
fn an_eliminated_row_spanning_the_range_solves_finite() {
    path_solves_as_its_scaled_rhs::<f32>(2f32.powi(56), 2f32.powi(-51), -120);
    path_solves_as_its_scaled_rhs::<f64>(2f64.powi(509), 2f64.powi(-485), -1010);
}
