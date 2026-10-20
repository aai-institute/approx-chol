#[path = "common/cases.rs"]
mod cases;
#[path = "common/dense.rs"]
mod dense;
#[path = "common/dimension.rs"]
mod dimension;
#[path = "common/entry.rs"]
mod entry;
#[path = "common/floats.rs"]
mod floats;
#[path = "common/graph.rs"]
mod graph;
#[path = "common/modes.rs"]
mod modes;
#[path = "common/residual.rs"]
mod residual;
#[path = "common/sparse.rs"]
mod sparse;

use approx_chol::{Config, Factor, InnerParallelism, Real, SchurMode, SolveError};
use cases::cases;
use entry::{by_type, cast, cast_all, Entry, Factored, Types};
use graph::{Edge, Graph};
use modes::Mode;
use sparse::Sparse;

fn assert_close(label: &str, x: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(x.len(), want.len(), "{label}");
    for (i, (&got, &want)) in x.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * want.abs().max(1.0),
            "{label}: x[{i}] = {got:e}, expected {want:e} ({x:?})"
        );
    }
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

/// Every vertex its own grounded block.
fn diagonal_sddm() -> Graph {
    Graph {
        ground: vec![2.0, 3.0, 5.0],
        ..Graph::new(3, Vec::new())
    }
}

cases! {
    /// A grounded singleton factors to `1/σ` under every config, so it solves `M⁻¹b` exactly.
    #[case::positive_rhs([1.0, 2.0, 3.0])]
    #[case::mixed_sign_rhs([1.0, -2.0, 4.0])]
    fn sddm_solve_matches_dense_inverse_nonzero_sum_rhs(
        #[case] b: [f64; 3],
        entry in entries(grounded),
        types in types(all),
    ) {
        let graph = diagonal_sddm();
        let reduced = entry.reduce(&graph, types);
        assert_eq!(reduced.floats(), [false; 3], "{types:?}");
        let factor = reduced.factorize(&Config::default());
        let want: Vec<f64> = b.iter().zip(&graph.ground).map(|(b, d)| b / d).collect();
        assert_close(&format!("{types:?}"), &factor.solve(&b), &want, 4.0 * types.epsilon());
    }
}

fn assert_buffers_checked<T: Real>(factor: &Factor<T>) {
    let n = factor.n();
    for found in [n - 1, n + 1] {
        let error = SolveError::LengthMismatch { expected: n, found };
        assert_eq!(factor.solve(&vec![T::zero(); found]), Err(error));
        let mut scratch = vec![T::zero(); factor.scratch_len()];
        let solved = factor.solve_in_place(&mut vec![T::zero(); found], &mut scratch);
        assert_eq!(solved, Err(error));
    }
    let needed = factor.scratch_len();
    let short = &mut vec![T::zero(); needed - 1];
    let error = SolveError::ScratchTooShort {
        needed,
        found: needed - 1,
    };
    assert_eq!(
        factor.solve_in_place(&mut vec![T::one(); n], short),
        Err(error)
    );
    let blocks = factor.clone().into_blocks();
    assert!(
        blocks.iter().any(|block| block.scratch_len() > 0),
        "a grounded block solves through scratch"
    );
    for block in blocks {
        let (m, needed) = (block.vertices().len(), block.scratch_len());
        let mut scratch = vec![T::zero(); needed];
        for found in [m - 1, m + 1] {
            let mut x = vec![T::zero(); found];
            let error = SolveError::LengthMismatch { expected: m, found };
            let solved = block.solve_local(&mut x, &mut scratch, InnerParallelism::Forbid);
            assert_eq!(solved, Err(error));
        }
        if needed > 0 {
            let mut x = vec![T::one(); m];
            let short = &mut scratch[..needed - 1];
            let error = SolveError::ScratchTooShort {
                needed,
                found: needed - 1,
            };
            let solved = block.solve_local(&mut x, short, InnerParallelism::Forbid);
            assert_eq!(solved, Err(error));
        }
    }
}

cases! {
    fn every_solve_checks_its_buffer_lengths(entry in entries(grounded), types in types(all)) {
        let graph = Graph {
            ground: vec![2.0, 3.0, 5.0],
            ..Graph::new(3, vec![Edge::negative(0, 1, 1.0)])
        };
        let reduced = entry.reduce(&graph, types);
        let factor = reduced.factorize(&Config::default());
        by_type!(&factor.factor, factor => assert_buffers_checked(factor));
    }
}

/// Each block through `solve_local` on a NaN-filled scratch, scattered back.
fn solve_on_dirty_scratch<T: Real>(
    factor: &Factor<T>,
    b: &[f64],
    inner: InnerParallelism,
) -> Vec<f64> {
    let mut x = vec![0.0; factor.n()];
    for block in factor.clone().into_blocks() {
        let mut local: Vec<T> = block.vertices().map(|v| cast(b[v])).collect();
        let mut scratch = vec![T::nan(); block.scratch_len() + 1];
        let solved = block.solve_local(&mut local, &mut scratch, inner);
        solved.expect("buffers fit");
        (block.vertices().zip(&local)).for_each(|(v, &value)| x[v] = cast(value));
    }
    x
}

fn uses_scratch(factor: &Factored) -> bool {
    by_type!(&factor.factor, factor => factor
        .clone()
        .into_blocks()
        .iter()
        .any(|block| block.scratch_len() > 0))
}

/// NaN reaches the result if any slot, ground or cover, is read before it is written.
fn assert_dirty_scratch_matches_the_solve(label: &str, factor: &Factored) {
    let n = factor.n();
    let rhs: Vec<f64> = (0..n).map(|i| i as f64 - 1.5).collect();
    let want = factor.solve(&rhs);
    for inner in [InnerParallelism::Allow, InnerParallelism::Forbid] {
        let local = factor.labels.to_local(&rhs);
        let x = by_type!(&factor.factor, f => solve_on_dirty_scratch(f, &local, inner));
        let x = factor.labels.to_caller(&x);
        assert_eq!(bits(&x), bits(&want), "{label} {inner:?}");
    }
    let local = factor.labels.to_local(&rhs);
    let x = by_type!(&factor.factor, f => solve_in_place_on_dirty_scratch(f, &local));
    let x = factor.labels.to_caller(&x);
    assert_eq!(bits(&x), bits(&want), "{label} in place");
}

/// The whole factor through `solve_in_place` on a NaN-filled scratch.
fn solve_in_place_on_dirty_scratch<T: Real>(factor: &Factor<T>, b: &[f64]) -> Vec<f64> {
    let mut x: Vec<T> = cast_all(b);
    let mut scratch = vec![T::nan(); factor.scratch_len() + 1];
    factor
        .solve_in_place(&mut x, &mut scratch)
        .expect("buffers fit");
    cast_all(&x)
}

cases! {
    fn dirty_scratch_does_not_change_a_grounded_solution(
        route in routes(grounded),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let graph = Graph {
            ground: vec![1.0, 1.0],
            ..Graph::new(2, vec![Edge::negative(0, 1, 1.0)])
        };
        let config = mode.config();
        let factor = entry.reduce(&graph, types).factorize(&config);
        assert!(
            uses_scratch(&factor),
            "grounded: the solve goes through scratch"
        );
        assert_dirty_scratch_matches_the_solve(&format!("grounded {types:?} {config:?}"), &factor);
    }
}

cases! {
    fn dirty_scratch_does_not_change_a_frustrated_solution(
        route in routes(frustrated),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let positive = |a, b| Edge {
            positive: true,
            ..Edge::negative(a, b, 1.0)
        };
        let graph = Graph::new(3, vec![positive(0, 1), positive(1, 2), positive(0, 2)]);
        let config = mode.config();
        let factor = entry.reduce(&graph, types).factorize(&config);
        assert!(
            uses_scratch(&factor),
            "frustrated: the solve goes through scratch"
        );
        let label = format!("frustrated {types:?} {config:?}");
        assert_dirty_scratch_matches_the_solve(&label, &factor);
    }
}

fn path(weight: f64) -> Graph {
    Graph::new(
        4,
        (0..3).map(|i| Edge::negative(i, i + 1, weight)).collect(),
    )
}

/// `t A` solves to `x / t`, so a factor that drops the scale misses by `t`.
fn assert_scaled_path_solves(entry: Entry, types: Types, mode: Mode, exponent: i32) {
    let graph = path(10f64.powi(exponent));
    let b = [1.0, 0.0, 0.0, -1.0];
    let config = mode.config();
    let x = entry.reduce(&graph, types).factorize(&config).solve(&b);
    let relative = graph.relative_residual(&x, &b);
    assert!(
        relative < 1e-6,
        "{types:?} {config:?} 1e{exponent}: relative residual {relative:e}"
    );
}

cases! {
    /// Each scalar runs the exponents its solution is representable at.
    fn a_scaled_f64_laplacian_solves_wherever_its_solution_is_representable(
        route in routes(all),
        types in types(f64),
        #[values(0, -5, -15, -100, -250)] exponent: i32,
    ) {
        let (entry, mode) = route;
        assert_scaled_path_solves(entry, types, mode, exponent);
    }
}

cases! {
    fn a_scaled_f32_laplacian_solves_wherever_its_solution_is_representable(
        route in routes(all),
        types in types(f32),
        #[values(0, -3, -10, -25)] exponent: i32,
    ) {
        let (entry, mode) = route;
        assert_scaled_path_solves(entry, types, mode, exponent);
    }
}

cases! {
    fn many_zero_singletons_factor_as_trivial_components(
        entry in entries(all),
        types in types(all),
    ) {
        let graph = Graph::new(128, Vec::new());
        let reduced = entry.reduce(&graph, types);
        assert!(
            reduced.floats().is_empty(),
            "{types:?}: a zero vertex is no block"
        );
        let factor = reduced.factorize(&Config::default());
        assert_eq!(factor.n(), 128);
        assert_eq!(bits(&factor.solve(&[1.0; 128])), bits(&[0.0; 128]));
        let x = by_type!(&factor.factor, f => solve_in_place_on_dirty_scratch(f, &[1.0; 128]));
        assert_eq!(bits(&x), bits(&[0.0; 128]), "{types:?}: in place zeroes every dead vertex");
    }
}

cases! {
    fn mixed_grounded_and_floating_components_solve_independently(
        route in routes(grounded),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let graph = Graph {
            ground: vec![2.0, 0.0, 0.0],
            ..Graph::new(3, vec![Edge::negative(1, 2, 1.0)])
        };
        let reduced = entry.reduce(&graph, types);
        let mut floats = reduced.floats();
        floats.sort();
        assert_eq!(floats, [false, true], "{types:?}");
        let config = mode.config();
        let x = reduced.factorize(&config).solve(&[4.0, 1.0, -1.0]);
        let label = format!("{types:?} {config:?}");
        assert_close(&label, &x, &[2.0, 0.5, -0.5], 4.0 * types.epsilon());
    }
}

cases! {
    /// Components `{0, 2}` and `{1, 3}` interleave; the inconsistent rhs solves least-squares.
    #[case::zero_sum_per_block([1.0, 2.0, -1.0, -2.0], [0.5, 1.0, -0.5, -1.0])]
    #[case::inconsistent([3.0, 5.0, -1.0, -2.0], [1.0, 1.75, -1.0, -1.75])]
    fn interleaved_components_solve_in_input_order(
        #[case] rhs: [f64; 4],
        #[case] want: [f64; 4],
        route in routes(all),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let edges = vec![Edge::negative(0, 2, 1.0), Edge::negative(1, 3, 1.0)];
        let graph = Graph::new(4, edges);
        let config = mode.config();
        let factor = entry.reduce(&graph, types).factorize(&config);
        let label = format!("{types:?} {config:?}");
        assert_close(&label, &factor.solve(&rhs), &want, 8.0 * types.epsilon());
    }
}

cases! {
    fn an_empty_system_solves_to_an_empty_vector(entry in entries(all), types in types(all)) {
        let reduced = entry.reduce(&Graph::new(0, Vec::new()), types);
        assert!(reduced.factorize(&Config::default()).solve(&[]).is_empty());
    }
}

#[test]
fn a_frustrated_block_with_a_negligible_surplus_solves_backward_stably() {
    use approx_chol::input::{SignedGraph, Weights};
    use approx_chol::{factorize_with, Backend};
    // A frustrated triangle, weakest edge 1e-80, and a pair on (1, 2) cancelling to a 2e-200 surplus.
    let (a, b) = ([0u32, 1, 0, 1, 1], [1u32, 2, 2, 2, 2]);
    let pos = [false, false, true, true, false];
    let w = [1e-80, 1.0, 2.0, 1e-200, 1e-200];
    let m = [
        [2.0 + 1e-80, -1e-80, 2.0],
        [-1e-80, 1.0 + 1e-80 + 2e-200, -1.0],
        [2.0, -1.0, 3.0 + 2e-200],
    ];
    let rhs = [2.5, -1.0, 3.5];
    let exact = Config {
        backend: Backend::ExactBelow { max_dim: 64 },
        complement: SchurMode::Exact,
        ..Config::default()
    };
    for config in [Config::default(), exact] {
        let graph = SignedGraph::new(3, &a, &b, &pos, Weights::of(&w).unwrap()).unwrap();
        let x = factorize_with(graph, config).unwrap().solve(&rhs).unwrap();
        let residual = (m.iter().zip(&rhs))
            .map(|(row, b)| (row.iter().zip(&x).map(|(m, x)| m * x).sum::<f64>() - b).abs())
            .fold(0.0, f64::max);
        let scale = 6.0 * x.iter().fold(0.0f64, |m, x| m.max(x.abs())) + 3.5;
        assert!(
            residual <= 64.0 * f64::EPSILON * scale,
            "{config:?}: x = {x:?}, residual {residual:e}"
        );
    }
}

#[test]
fn a_grounded_path_behind_a_light_edge_solves_on_the_approximate_arm() {
    use approx_chol::input::{Laplacian, Weights};
    use approx_chol::{factorize_with, Backend};
    // Min-degree eliminates the ground first and leaves vertex 0 as the sink; `−Σb` then cancels to 1e24 over a 1e-290 pivot.
    let (a, b, w, ground) = ([0u32, 1], [1u32, 2], [1e-290, 1e30], [0.0, 1e40, 0.0]);
    let x = [1.0, 0.75, -0.5];
    let rhs = [2.5e-291, 0.75e40 + 1.25e30, -1.25e30];
    let config = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };
    let path = Laplacian::new(3, &a, &b, Weights::of(&w).unwrap()).unwrap();
    let factor = factorize_with(path.grounded(&ground).unwrap(), config).unwrap();
    assert_close(
        "light sink",
        &factor.solve(&rhs).unwrap(),
        &x,
        64.0 * f64::EPSILON,
    );
}

#[test]
fn ac2_on_a_one_column_grounded_complement_solves_exactly() {
    use approx_chol::input::{BipartiteGram, FactorWidth1, Weights};
    use approx_chol::{factorize_with, Backend};
    // Three left vertices on one right vertex, each by a cancelling pair: `S` is one grounded slot, so AC2 samples no edge.
    let (left, right) = ([0u32, 0, 1, 1, 2, 2], [0u32; 6]);
    let loadings = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let weights = [2.0, 1.0, 3.0, 1.0, 1.0, 4.0];
    let x = [1.0, -1.0, 2.0, 0.5];
    let b = [3.5, -3.0, 8.5, -1.0];
    for k in [2, 3] {
        for complement in [SchurMode::Same, SchurMode::Exact] {
            let config = Config {
                split_merge: Some(k),
                backend: Backend::Approximate,
                complement,
                ..Config::default()
            };
            let gram = BipartiteGram::new(
                FactorWidth1::new(3, &left, Some(&loadings[..])).unwrap(),
                FactorWidth1::<f64>::new(1, &right, None).unwrap(),
                Weights::of(&weights).unwrap(),
            )
            .unwrap();
            let solved = factorize_with(gram, config).unwrap().solve(&b).unwrap();
            assert_close(&format!("{config:?}"), &solved, &x, 64.0 * f64::EPSILON);
        }
    }
}
