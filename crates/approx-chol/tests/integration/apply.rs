use std::array;

use approx_chol::input::{CsrRef, Laplacian};
use approx_chol::{
    factorize, factorize_with, ApplyError, ApproxInverse, ColumnMajor, Columns, Config,
    FrustratedRoute, LayoutError, MatrixMut, MemStack, Real, ScalingPolicy, StackReq, Threads,
};

use crate::common::apart::Apart;
use crate::common::bits::{bits, canonical_bits};
use crate::common::cases::cases;
use crate::common::entry::{by_type, cast_all, Entry, Factored, Labels, Types};
use crate::common::graph::{Edge, Graph};
use crate::common::layouts::{assert_errors_leave_the_block, Layout, Layouts};
use crate::common::modes::Mode;
use crate::common::scratch::{nan_cells, nan_slots};
use crate::common::sparse::Sparse;
use crate::common::threads;

/// How the odd vertices `1, 3, 5` join.
#[derive(Clone, Copy)]
enum Triangle {
    Negative,
    Positive,
    Open,
}

/// Even vertices a 4-cycle and a tail, listed; `1, 3, 5` per `triangle`; `12..16` a path, a range; `7, 9, 11` alone, isolated where `grounded`, else dead.
fn components(grounded: bool, triangle: Triangle) -> Graph {
    let mut edges: Vec<Edge> = [(0, 2), (2, 4), (4, 6), (0, 6), (6, 8), (8, 10)]
        .map(|(a, b)| Edge::negative(a, b, 1.0 + a as f64 / 8.0))
        .to_vec();
    edges.extend(match triangle {
        Triangle::Negative => [(1, 3), (3, 5), (1, 5)]
            .map(|(a, b)| Edge::negative(a, b, 1.5))
            .to_vec(),
        Triangle::Positive => [(1, 3), (3, 5), (1, 5)]
            .map(|(a, b)| Edge::positive(a, b, 1.5))
            .to_vec(),
        Triangle::Open => [(1, 3), (3, 5)]
            .map(|(a, b)| Edge::negative(a, b, 1.5))
            .to_vec(),
    });
    edges.extend([(12, 13), (13, 14), (14, 15)].map(|(a, b)| Edge::negative(a, b, 0.75)));
    let mut graph = Graph::new(16, edges);
    if grounded {
        for (v, g) in [
            (0, 0.5),
            (3, 0.25),
            (13, 1.0),
            (7, 2.0),
            (9, 0.5),
            (11, 4.0),
        ] {
            graph.ground[v] = g;
        }
    }
    graph
}

/// Column `j` of the right-hand sides in the caller's numbering; column 2 is zero, and no two others are equal.
fn rhs(n: usize, j: usize) -> Vec<f64> {
    let value = |i: usize| match j {
        2 => 0.0,
        _ => ((i * 7 + j * 13) % 11) as f64 - 5.0 + (j / 11) as f64 * 0.5,
    };
    (0..n).map(value).collect()
}

/// Each column of a `K`-column block in the caller's numbering, per layout, on a NaN-filled stack of `scratch(K, threads)` and on one of exactly the slots its layout takes at one thread.
fn applied<T: Real, const K: usize>(
    factor: &ApproxInverse<T>,
    labels: &Labels,
    threads: Threads,
) -> Vec<(String, Vec<Vec<f64>>)> {
    let n = factor.n();
    let local = |j| cast_all(&labels.to_local(&rhs(n, j)));
    let fresh = Layouts::<T, K>::new(array::from_fn(local));
    let scratch = factor.scratch(K, threads).size_bytes() / size_of::<T>();
    let mut solved = Vec::new();
    for &layout in Layout::ALL {
        let least = fresh.least(factor, layout);
        for (stack, slots) in [("scratch", scratch + 1), ("its least", least)] {
            let mut block = fresh.clone();
            let mut cells = nan_slots::<T>(slots);
            block
                .on(layout, |b| {
                    b.apply_with(factor, threads, MemStack::new_any(&mut cells))
                })
                .expect("its own scratch");
            let columns = block.columns(layout).into_iter();
            let caller = columns.map(|column| labels.to_caller(&cast_all(&column)));
            solved.push((format!("{layout:?} on {stack}"), caller.collect()));
        }
    }
    solved
}

/// Every column of every block width and budget equals that column's solve alone, bit for bit.
fn assert_columns_solve_alone(label: &str, factored: &Factored, types: Types) {
    let n = by_type!(&factored.factor, f => f.n());
    let alone: Vec<Vec<f64>> = (0..17).map(|j| factored.solve(&rhs(n, j))).collect();
    let reached = alone[0].iter().any(|x| x.abs() > types.epsilon());
    assert!(reached, "{label}: the right-hand side reaches the solution");
    by_type!(&factored.factor, f => assert_blocks(label, f, &factored.labels, &alone));
}

fn assert_blocks<T: Real>(
    label: &str,
    factor: &ApproxInverse<T>,
    labels: &Labels,
    alone: &[Vec<f64>],
) {
    for threads in threads::BUDGETS {
        // Every remainder a pack leaves, alone and after one whole pack, and past two.
        let blocks = [
            applied::<T, 1>(factor, labels, threads),
            applied::<T, 2>(factor, labels, threads),
            applied::<T, 3>(factor, labels, threads),
            applied::<T, 4>(factor, labels, threads),
            applied::<T, 5>(factor, labels, threads),
            applied::<T, 6>(factor, labels, threads),
            applied::<T, 7>(factor, labels, threads),
            applied::<T, 8>(factor, labels, threads),
            applied::<T, 9>(factor, labels, threads),
            applied::<T, 10>(factor, labels, threads),
            applied::<T, 11>(factor, labels, threads),
            applied::<T, 12>(factor, labels, threads),
            applied::<T, 13>(factor, labels, threads),
            applied::<T, 14>(factor, labels, threads),
            applied::<T, 15>(factor, labels, threads),
            applied::<T, 16>(factor, labels, threads),
            applied::<T, 17>(factor, labels, threads),
        ];
        for (layout, block) in blocks.iter().flatten() {
            for (j, column) in block.iter().enumerate() {
                let k = block.len();
                let (column, alone) = (bits(column), bits(&alone[j]));
                assert_eq!(
                    column, alone,
                    "{label}: {layout} column {j} of {k}, {threads:?}"
                );
            }
        }
    }
    let mut x: Vec<T> = cast_all(&labels.to_local(&rhs(factor.n(), 0)));
    factor
        .apply(x.as_mut_slice(), Threads::All)
        .expect("n rows");
    let x = labels.to_caller(&cast_all(&x));
    assert_eq!(
        bits(&x),
        bits(&alone[0]),
        "{label}: apply allocates its own scratch"
    );
}

cases! {
    fn grounded_blocks_solve_each_column_alone(route in routes(grounded), types in types(all)) {
        let (entry, mode) = route;
        let reduced = entry.reduce(&components(true, Triangle::Negative), types);
        assert!(reduced.floats().contains(&false), "a block is grounded");
        let factor = reduced.factorize(&mode.config());
        let label = format!("grounded {entry:?} {types:?} {mode:?}");
        assert_columns_solve_alone(&label, &factor, types);
        by_type!(&factor.factor, f => assert_errors_leave_the_block(f));
    }
}

cases! {
    fn frustrated_blocks_solve_each_column_alone(route in routes(frustrated), types in types(all)) {
        let (entry, mode) = route;
        let reduced = entry.reduce(&components(false, Triangle::Positive), types);
        for frustrated in [mode.config().frustrated, FrustratedRoute::Signed] {
            let factor = reduced.factorize(&Config { frustrated, ..mode.config() });
            let label = format!("frustrated {entry:?} {types:?} {mode:?} {frustrated:?}");
            assert_columns_solve_alone(&label, &factor, types);
        }
    }
}

cases! {
    fn floating_blocks_solve_each_column_alone(
        route in routes(floating_non_bipartite),
        types in types(all),
    ) {
        let (entry, mode) = route;
        let reduced = entry.reduce(&components(false, Triangle::Negative), types);
        assert!(reduced.floats().iter().all(|&floats| floats), "every block floats");
        let factor = reduced.factorize(&mode.config());
        let label = format!("floating {entry:?} {types:?} {mode:?}");
        assert_columns_solve_alone(&label, &factor, types);
    }
}

cases! {
    fn bipartite_blocks_solve_each_column_alone(route in routes(floating), types in types(all)) {
        let (entry, mode) = route;
        let factor = entry.reduce(&components(false, Triangle::Open), types).factorize(&mode.config());
        let label = format!("bipartite {entry:?} {types:?} {mode:?}");
        assert_columns_solve_alone(&label, &factor, types);
    }
}

/// Claims more columns than any scratch holds.
struct Endless(usize);

impl MatrixMut<f64> for Endless {
    fn rows(&self) -> usize {
        self.0
    }

    fn columns(&self) -> usize {
        usize::MAX
    }

    fn read_column(&self, _: usize, _: &mut [f64]) {
        unreachable!("an apply refused reads no column");
    }

    fn write_column(&mut self, _: usize, _: &[f64]) {
        unreachable!("an apply refused writes no column");
    }
}

/// Scratch for more columns than memory holds overflows; a copied block of that many takes one pack's, so it errs on a stack short of that, before it reads a column.
#[test]
fn a_block_of_more_columns_than_memory_errs_on_a_short_stack() {
    let interleaved = Laplacian::new(6, &[0u32, 2, 1, 3], &[2, 4, 3, 5]).unwrap();
    let factor: ApproxInverse = factorize(interleaved).unwrap();
    let past = factor.scratch(usize::MAX, Threads::All);
    assert_eq!(past.align_bytes(), 0, "the requirement overflows");
    let mut cells = nan_slots::<f64>(1);
    for slots in [0, 1] {
        let short = MemStack::new_any(&mut cells[..slots]);
        let error = factor.apply_with(&mut Endless(6), Threads::All, short);
        assert_eq!(error, Err(ApplyError::Scratch), "{slots} slots");
    }
    let error = factor.apply(&mut Endless(5), Threads::All);
    assert_eq!(
        error,
        Err(ApplyError::Rows {
            expected: 6,
            found: 5
        })
    );
}

/// A copied block's stack stops growing at a pack: 45 columns take no more than 17, where `scratch` for them is larger and `[T; K]` takes all `K` at once.
#[test]
fn a_copied_block_takes_one_packs_stack_whatever_its_width() {
    let interleaved = Laplacian::new(6, &[0u32, 2, 1, 3], &[2, 4, 3, 5]).unwrap();
    let factor: ApproxInverse = factorize(interleaved).unwrap();
    let wide = Layouts::<f64, 45>::new(array::from_fn(|j| rhs(6, j)));
    let narrow = Layouts::<f64, 17>::new(array::from_fn(|j| rhs(6, j)));
    let alone: Vec<Vec<u64>> = (0..45)
        .map(|j| {
            let mut x = rhs(6, j);
            factor.apply(&mut x, Threads::All).expect("6 rows");
            bits(&x)
        })
        .collect();
    let scratch = factor.scratch(45, threads::ONE).size_bytes() / size_of::<f64>();
    let copied = [
        Layout::Apart,
        Layout::ColumnMajor,
        Layout::Columns,
        #[cfg(feature = "nalgebra")]
        Layout::Nalgebra,
        #[cfg(feature = "ndarray")]
        Layout::NdarrayRowMajor,
        #[cfg(feature = "ndarray")]
        Layout::NdarrayColumnMajor,
    ];
    for layout in copied {
        let least = wide.least(&factor, layout);
        assert!(least <= narrow.least(&factor, layout), "{layout:?}");
        assert!(least < scratch, "{layout:?}");
        for threads in threads::BUDGETS {
            let mut block = wide.clone();
            let mut cells = nan_slots::<f64>(least);
            block
                .on(layout, |b| {
                    b.apply_with(&factor, threads, MemStack::new_any(&mut cells))
                })
                .expect("its own stack");
            let columns: Vec<Vec<u64>> = block.columns(layout).iter().map(|c| bits(c)).collect();
            assert_eq!(columns, alone, "{layout:?} at {threads:?}");
        }
    }
    let interleaved = wide.least(&factor, Layout::Interleaved);
    for layout in copied {
        assert!(interleaved > wide.least(&factor, layout), "{layout:?}");
    }
}

/// Fifteen columns take a pack of eight, then seven in a pack of eight whose spare lane keeps column 7's solution: NaN or inf there leaves columns 8 to 14 as they solve alone.
#[test]
fn a_column_solved_to_nan_reaches_no_column_of_the_next_pack() {
    let interleaved = Laplacian::new(6, &[0u32, 2, 1, 3], &[2, 4, 3, 5]).unwrap();
    let factor: ApproxInverse = factorize(interleaved).unwrap();
    for poison in [f64::NAN, f64::INFINITY] {
        let column = |j| {
            let mut b = rhs(6, j);
            b[0] = if j == 7 { poison } else { b[0] };
            b
        };
        let fresh = Layouts::<f64, 15>::new(array::from_fn(column));
        let alone: Vec<Vec<u64>> = (8..15)
            .map(|j| {
                let mut x = column(j);
                factor.apply(&mut x, Threads::All).expect("6 rows");
                bits(&x)
            })
            .collect();
        for &layout in Layout::ALL {
            for threads in [threads::ONE, Threads::All] {
                let mut block = fresh.clone();
                block
                    .on(layout, |b| b.apply(&factor, threads))
                    .expect("6 rows");
                let columns = block.columns(layout);
                assert!(columns[7].iter().any(|x| !x.is_finite()), "{layout:?}");
                let after: Vec<Vec<u64>> = columns[8..].iter().map(|c| bits(c)).collect();
                assert_eq!(after, alone, "{layout:?} {poison} at {threads:?}");
            }
        }
    }
}

/// An empty stack's slice is not aligned for `f64`: an apply with nothing to solve takes no slot from it.
#[test]
fn an_apply_with_nothing_to_solve_accepts_an_empty_stack() {
    let path: ApproxInverse = factorize(Laplacian::new(3, &[0u32, 1], &[1, 2]).unwrap()).unwrap();
    assert_eq!(path.scratch(0, Threads::All), StackReq::EMPTY);
    let mut no_columns: Vec<[f64; 0]> = vec![[]; 3];
    path.apply_with(&mut no_columns, Threads::All, MemStack::new(&mut []))
        .unwrap();
    let factor: ApproxInverse = factorize(Laplacian::new(0, &[0u32; 0], &[]).unwrap()).unwrap();
    assert_eq!(factor.scratch(3, Threads::All), StackReq::EMPTY);
    let mut apart = Apart(vec![Vec::<f64>::new(); 3]);
    factor
        .apply_with(&mut apart, Threads::All, MemStack::new(&mut []))
        .unwrap();
    let mut rows: Vec<[f64; 3]> = Vec::new();
    factor
        .apply_with(&mut rows, Threads::All, MemStack::new(&mut []))
        .unwrap();
    let mut empty = vec![Vec::<f64>::new(); 3];
    let mut slices = Columns::new(&mut empty).unwrap();
    factor
        .apply_with(&mut slices, Threads::All, MemStack::new(&mut []))
        .unwrap();
    for factor in [&factor, &path] {
        let mut major = ColumnMajor::new(&mut [] as &mut [f64], 0).unwrap();
        assert_eq!(major.columns(), 0);
        factor
            .apply_with(&mut major, Threads::All, MemStack::new(&mut []))
            .unwrap();
    }
}

/// Past the spread's threshold in small blocks: 300 paths of 3 to 32 vertices interleaved in the numbering, listed, the short ones dense; 200 contiguous paths of 25, runs; 50 contiguous 12-cycles with one positive edge, covers; then a 120×120 grid, which takes the whole budget alone.
fn many_components() -> ApproxInverse {
    let lens: Vec<usize> = (0..300).map(|c| 3 + c % 30).collect();
    let mut ids: Vec<Vec<usize>> = vec![Vec::new(); lens.len()];
    let mut next = 0;
    for j in 0..32 {
        for (c, &len) in lens.iter().enumerate() {
            if j < len {
                ids[c].push(next);
                next += 1;
            }
        }
    }
    let mut edges: Vec<Edge> = (ids.iter())
        .flat_map(|path| path.windows(2).map(|p| Edge::negative(p[0], p[1], 1.0)))
        .collect();
    for _ in 0..200 {
        edges.extend((next..next + 24).map(|v| Edge::negative(v, v + 1, 1.5)));
        next += 25;
    }
    for _ in 0..50 {
        edges.extend((next..next + 11).map(|v| Edge::negative(v, v + 1, 2.0)));
        edges.push(Edge::positive(next, next + 11, 1.0));
        next += 12;
    }
    let side = 120;
    for v in 0..side * side {
        if v % side + 1 < side {
            edges.push(Edge::negative(next + v, next + v + 1, 1.0));
        }
        if v + side < side * side {
            edges.push(Edge::negative(next + v, next + v + side, 1.0));
        }
    }
    let graph = Graph::new(next + side * side, edges);
    let (ptrs, columns, values) = graph.csr(false);
    let csr = CsrRef::new(graph.n, &ptrs, &columns, &values).unwrap();
    let config = Config {
        exact_max_dim: 8,
        ..Config::default()
    };
    factorize_with(csr, config, ScalingPolicy::default()).unwrap()
}

/// Every budget, on a stack of one region or of more, solves each column to one thread's bits.
#[test]
fn a_budget_spread_over_many_components_changes_no_bit() {
    let factor = many_components();
    assert_eq!(factor.components().len(), 551);
    assert!(
        factor.n_steps() - 120 * 120 > 8192,
        "small blocks past the spread's threshold"
    );
    assert_spread_changes_no_bit(&factor);
}

/// A decoded factor's segments come from its payload, not the build: they spread as the built ones do.
#[cfg(feature = "serde")]
#[test]
fn a_decoded_factor_spreads_to_the_same_bits() {
    let factor = many_components();
    let bytes = postcard::to_stdvec(&factor).expect("serialize");
    let decoded: ApproxInverse = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(
        spread_bits::<9>(&decoded, Threads::All, Threads::All),
        spread_bits::<9>(&factor, threads::ONE, threads::ONE)
    );
    assert_spread_changes_no_bit(&decoded);
}

/// `K` columns in each layout applied at `threads` on a NaN-filled stack of `scratch(K, regions)`.
fn spread_bits<const K: usize>(
    factor: &ApproxInverse,
    threads: Threads,
    regions: Threads,
) -> Vec<(Layout, Vec<Vec<u64>>)> {
    let fresh = Layouts::<f64, K>::new(array::from_fn(|j| rhs(factor.n(), j)));
    let mut cells = nan_cells::<f64>(factor.scratch(K, regions));
    let solve = |&layout: &Layout| {
        let mut block = fresh.clone();
        block
            .on(layout, |b| {
                b.apply_with(factor, threads, MemStack::new_any(&mut cells))
            })
            .expect("one region at least");
        (
            layout,
            block.columns(layout).iter().map(|c| bits(c)).collect(),
        )
    };
    Layout::ALL.iter().map(solve).collect()
}

/// Every layout at every budget, on a stack of one region or of more, solves each column to the bits of the interleaved block on one thread: a pack and a column alone, and a pack and each remainder of two to seven.
fn assert_spread_changes_no_bit(factor: &ApproxInverse) {
    assert_spread_at::<9>(factor);
    assert_spread_at::<10>(factor);
    assert_spread_at::<11>(factor);
    assert_spread_at::<13>(factor);
    assert_spread_at::<14>(factor);
    assert_spread_at::<15>(factor);
}

fn assert_spread_at<const K: usize>(factor: &ApproxInverse) {
    let one = spread_bits::<K>(factor, threads::ONE, threads::ONE);
    let (Layout::Interleaved, alone) = &one[0] else {
        unreachable!("the interleaved block comes first")
    };
    let eight = Threads::Count(std::num::NonZeroUsize::new(8).unwrap());
    for threads in threads::BUDGETS.into_iter().chain([eight]) {
        for regions in [threads::ONE, threads::BUDGETS[2], threads] {
            for (layout, columns) in spread_bits::<K>(factor, threads, regions) {
                assert_eq!(columns.len(), alone.len());
                for (j, (column, alone)) in columns.iter().zip(alone).enumerate() {
                    assert_eq!(
                        column, alone,
                        "{layout:?} column {j} of {K}, {threads:?} on {regions:?}'s regions"
                    );
                }
            }
        }
    }
}

/// Each thread of the budget asks a region of its own, and the copied column comes once.
#[test]
fn scratch_grows_with_the_budget() {
    let factor = many_components();
    let three = threads::BUDGETS[2];
    for columns in [1, 4] {
        let (one, more) = (
            factor.scratch(columns, threads::ONE),
            factor.scratch(columns, three),
        );
        assert!(more.size_bytes() > one.size_bytes(), "{columns} columns");
    }
}

/// Each malformed layout is refused where it is formed, naming what is wrong.
#[test]
fn a_malformed_layout_errs_where_it_is_formed() {
    let mut data = [0.0f64; 6];
    for rows in [4, 0] {
        assert_eq!(
            ColumnMajor::new(&mut data, rows).err(),
            Some(LayoutError::Length { rows, found: 6 })
        );
    }
    let mut columns = [vec![0.0f64; 3], vec![0.0; 2], vec![0.0; 3]];
    assert_eq!(
        Columns::new(&mut columns).err(),
        Some(LayoutError::Column {
            column: 1,
            expected: 3,
            found: 2
        })
    );
}

/// A block without columns is left as it is, whatever its rows; with one, its rows are the operator's.
#[test]
fn only_a_block_with_columns_is_held_to_the_rows() {
    let path: ApproxInverse = factorize(Laplacian::new(3, &[0u32, 1], &[1, 2]).unwrap()).unwrap();
    let mut no_columns: Vec<Vec<f64>> = Vec::new();
    let mut empty: [f64; 0] = [];
    let mut slices = Columns::new(&mut no_columns).unwrap();
    let mut major = ColumnMajor::new(&mut empty, 2).unwrap();
    let mut rows: Vec<[f64; 0]> = vec![[]; 2];
    for block in [
        &mut slices as &mut dyn MatrixMut<f64>,
        &mut major,
        &mut rows,
    ] {
        assert_eq!(block.columns(), 0);
        path.apply_with(block, Threads::All, MemStack::new(&mut []))
            .expect("nothing to solve");
        path.apply(block, Threads::All).expect("nothing to solve");
    }
    let mut data = [1.0, -1.0, 2.0, -2.0];
    let mut major = ColumnMajor::new(&mut data, 2).unwrap();
    assert_eq!(
        path.apply(&mut major, Threads::All),
        Err(ApplyError::Rows {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(data, [1.0, -1.0, 2.0, -2.0], "left as it was");
}

/// Through the trait, each layout copies one column in and out and touches no other.
#[test]
fn a_layout_reads_and_writes_one_column() {
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut major = ColumnMajor::new(&mut data, 3).unwrap();
    assert_eq!((major.rows(), major.columns()), (3, 2));
    let mut column = [0.0; 3];
    major.read_column(1, &mut column);
    assert_eq!(column, [4.0, 5.0, 6.0]);
    major.write_column(0, &[7.0, 8.0, 9.0]);
    assert_eq!(data, [7.0, 8.0, 9.0, 4.0, 5.0, 6.0]);
    let mut columns = [[1.0, 2.0], [3.0, 4.0]];
    let mut slices = Columns::new(&mut columns).unwrap();
    assert_eq!((slices.rows(), slices.columns()), (2, 2));
    let mut column = [0.0; 2];
    slices.read_column(1, &mut column);
    assert_eq!(column, [3.0, 4.0]);
    slices.write_column(0, &[5.0, 6.0]);
    assert_eq!(columns, [[5.0, 6.0], [3.0, 4.0]]);
}

/// A column past the block panics.
#[test]
#[should_panic(expected = "column 2 of 2")]
fn a_column_major_column_past_the_block_panics() {
    let mut data = [0.0f64; 4];
    let major = ColumnMajor::new(&mut data, 2).unwrap();
    major.read_column(2, &mut [0.0; 2]);
}

/// A floating bipartite Gram its congruence negates from vertex 4, `b` past `f64::MAX`: the projection's sum is NaN, which the negated rows reach; each column is NaN where it is alone, and bit for bit elsewhere (#437).
#[test]
fn a_nan_reaching_a_negated_row_is_nan_in_the_same_rows_in_a_block_as_alone() {
    let g = crate::common::bipartite::Bipartite {
        n_left: 4,
        n_right: 3,
        left: vec![0, 0, 1, 1, 2, 2, 3, 3],
        right: vec![0, 1, 1, 2, 0, 2, 0, 1],
        values: None,
    };
    let factor = factorize(g.view()).unwrap();
    assert!(
        format!("{factor:?}").contains("NegateFrom(4)"),
        "{factor:?}"
    );
    let b: Vec<f64> = (0..7).map(|i| (i as f64 - 3.0) * 6e307).collect();
    let mut rows: Vec<[f64; 3]> = b.iter().map(|&v| [v, -v, 0.5 * v]).collect();
    factor.apply(rows.as_mut_slice(), Threads::All).unwrap();
    let mut nan = false;
    for (j, sign) in [1.0, -1.0, 0.5].into_iter().enumerate() {
        let mut alone: Vec<f64> = b.iter().map(|&v| sign * v).collect();
        factor.apply(&mut alone, Threads::All).unwrap();
        let column: Vec<f64> = rows.iter().map(|row| row[j]).collect();
        assert_eq!(
            canonical_bits(&column),
            canonical_bits(&alone),
            "column {j}"
        );
        nan |= alone[4..].iter().any(|x| x.is_nan());
    }
    assert!(nan, "a NaN reaches a negated row");
}
