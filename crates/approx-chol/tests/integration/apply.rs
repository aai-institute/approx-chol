use std::array;

use approx_chol::input::Laplacian;
use approx_chol::{
    factorize, factorize_with, ApplyError, ApproxInverse, ColumnMajor, Columns, Config, CsrRef,
    LayoutError, MatrixMut, MemStack, Real, ScalingPolicy, StackReq, Threads,
};

use crate::common::bits::bits;
use crate::common::cases::cases;
use crate::common::entry::{by_type, cast, cast_all, Entry, Factored, Labels, Types};
use crate::common::graph::{Edge, Graph};
use crate::common::modes::Mode;
use crate::common::scratch::nan_cells;
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

/// Column `j` of the right-hand sides in the caller's numbering; column 2 is zero.
fn rhs(n: usize, j: usize) -> Vec<f64> {
    let value = |i: usize| match j {
        2 => 0.0,
        _ => ((i * 7 + j * 13) % 11) as f64 - 5.0,
    };
    (0..n).map(value).collect()
}

/// Columns stored apart, so the apply copies them through its scratch.
struct Apart<T>(Vec<Vec<T>>);

impl<T: Real> MatrixMut<T> for Apart<T> {
    fn rows(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }

    fn columns(&self) -> usize {
        self.0.len()
    }

    fn read_column(&self, column: usize, into: &mut [T]) {
        into.copy_from_slice(&self.0[column]);
    }

    fn write_column(&mut self, column: usize, from: &[T]) {
        self.0[column].copy_from_slice(from);
    }
}

/// Each column of a `K`-column block on a NaN-filled stack in the caller's numbering, per layout.
fn applied<T: Real, const K: usize>(
    factor: &ApproxInverse<T>,
    labels: &Labels,
    threads: Threads,
) -> [(&'static str, Vec<Vec<f64>>); 4] {
    let n = factor.n();
    let columns: [Vec<T>; K] = array::from_fn(|j| cast_all(&labels.to_local(&rhs(n, j))));
    let mut rows: Vec<[T; K]> = (0..n).map(|i| array::from_fn(|j| columns[j][i])).collect();
    let mut apart = Apart(columns.to_vec());
    let mut major = columns.concat();
    let mut slices = columns.to_vec();
    {
        let mut major = ColumnMajor::new(&mut major, n).expect("n rows to a column");
        let mut slices = Columns::new(&mut slices).expect("n rows each");
        for block in [
            &mut rows as &mut dyn MatrixMut<T>,
            &mut apart,
            &mut major,
            &mut slices,
        ] {
            let mut cells = nan_cells::<T>(factor.scratch(K, threads));
            let stack = MemStack::new_any(&mut cells);
            factor
                .apply_with(block, threads, stack)
                .expect("its own scratch");
        }
    }
    let caller = |column: &[T]| labels.to_caller(&cast_all(column));
    let interleaved = (0..K).map(|j| caller(&rows.iter().map(|row| row[j]).collect::<Vec<_>>()));
    [
        ("interleaved", interleaved.collect()),
        ("apart", apart.0.iter().map(|c| caller(c)).collect()),
        (
            "column-major",
            (0..K).map(|j| caller(&major[j * n..][..n])).collect(),
        ),
        ("columns", slices.iter().map(|c| caller(c)).collect()),
    ]
}

/// Every column of every block width and budget equals that column's solve alone, bit for bit.
fn assert_columns_solve_alone(label: &str, factored: &Factored, types: Types) {
    let n = by_type!(&factored.factor, f => f.n());
    let alone: Vec<Vec<f64>> = (0..8).map(|j| factored.solve(&rhs(n, j))).collect();
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
        let blocks = [
            applied::<T, 1>(factor, labels, threads),
            applied::<T, 2>(factor, labels, threads),
            applied::<T, 3>(factor, labels, threads),
            applied::<T, 4>(factor, labels, threads),
            applied::<T, 8>(factor, labels, threads),
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

/// A block of the wrong row count, or a stack short of one thread's scratch, errs and leaves the block as it was, in every layout.
fn assert_errors_leave_the_block<T: Real>(factor: &ApproxInverse<T>) {
    let n = factor.n();
    let needed = factor.scratch(2, threads::ONE);
    let mut cells = nan_cells::<T>(needed);
    let short = needed.size_bytes() / size_of::<T>() - 1;
    for found in [n - 1, n + 1, n] {
        let column = |sign: f64| {
            (0..found)
                .map(|i| cast(sign * i as f64))
                .collect::<Vec<T>>()
        };
        let columns = [column(1.0), column(-1.0)];
        let mut rows: Vec<[T; 2]> = (0..found).map(|i| [columns[0][i], columns[1][i]]).collect();
        let mut apart = Apart(columns.to_vec());
        let mut data = columns.concat();
        let mut slices = columns.to_vec();
        {
            let mut major = ColumnMajor::new(&mut data, found).expect("found rows to a column");
            let mut slices = Columns::new(&mut slices).expect("found rows each");
            for block in [
                &mut rows as &mut dyn MatrixMut<T>,
                &mut apart,
                &mut major,
                &mut slices,
            ] {
                match found == n {
                    true => {
                        let stack = MemStack::new_any(&mut cells[..short]);
                        let error = factor.apply_with(block, Threads::All, stack);
                        assert_eq!(error, Err(ApplyError::Scratch));
                    }
                    false => {
                        let rows = Err(ApplyError::Rows { expected: n, found });
                        assert_eq!(factor.apply(block, Threads::All), rows);
                        let stack = MemStack::new_any(&mut cells);
                        assert_eq!(factor.apply_with(block, Threads::All, stack), rows);
                    }
                }
            }
        }
        let before = columns.concat();
        let interleaved = (0..found).map(|i| [before[i], before[found + i]]);
        assert!(
            rows == interleaved.collect::<Vec<_>>(),
            "interleaved of {found} rows"
        );
        for (layout, after) in [
            ("apart", apart.0.concat()),
            ("column-major", data),
            ("columns", slices.concat()),
        ] {
            assert!(
                after == before,
                "{layout} of {found} rows is left as it was"
            );
        }
    }
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
        let factor = entry.reduce(&components(false, Triangle::Positive), types).factorize(&mode.config());
        let label = format!("frustrated {entry:?} {types:?} {mode:?}");
        assert_columns_solve_alone(&label, &factor, types);
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
        unreachable!("an apply past memory reads no column");
    }

    fn write_column(&mut self, _: usize, _: &[f64]) {
        unreachable!("an apply past memory writes no column");
    }
}

#[test]
fn a_block_past_memory_errs_on_its_scratch() {
    let interleaved = Laplacian::new(6, &[0u32, 2, 1, 3], &[2, 4, 3, 5]).unwrap();
    let factor: ApproxInverse = factorize(interleaved).unwrap();
    let past = factor.scratch(usize::MAX, Threads::All);
    assert_eq!(past.align_bytes(), 0, "the requirement overflows");
    assert_eq!(
        factor.apply(&mut Endless(6), Threads::All),
        Err(ApplyError::Scratch)
    );
    let empty = MemStack::new(&mut []);
    let error = factor.apply_with(&mut Endless(6), Threads::All, empty);
    assert_eq!(error, Err(ApplyError::Scratch));
    let error = factor.apply(&mut Endless(5), Threads::All);
    assert_eq!(
        error,
        Err(ApplyError::Rows {
            expected: 6,
            found: 5
        })
    );
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
    let csr = CsrRef::new(&ptrs, &columns, &values, graph.n as u32).unwrap();
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
        spread_bits(&decoded, Threads::All, Threads::All),
        spread_bits(&factor, threads::ONE, threads::ONE)
    );
    assert_spread_changes_no_bit(&decoded);
}

/// Two columns interleaved, apart through the scratch, and column-major and one slice each where they are.
fn spread_bits(factor: &ApproxInverse, threads: Threads, regions: Threads) -> [Vec<u64>; 4] {
    let n = factor.n();
    let column = |j: usize| -> Vec<f64> {
        (0..n)
            .map(|i| ((i * 7 + j * 5) % 11) as f64 - 5.0)
            .collect()
    };
    let (zero, one) = (column(0), column(1));
    let mut rows: Vec<[f64; 2]> = (zero.iter().zip(&one)).map(|(&a, &b)| [a, b]).collect();
    let mut data = [zero.as_slice(), &one].concat();
    let mut slices = vec![zero.clone(), one.clone()];
    let mut apart = Apart(vec![zero, one]);
    {
        let mut major = ColumnMajor::new(&mut data, n).expect("n rows to a column");
        let mut slices = Columns::new(&mut slices).expect("n rows each");
        for block in [
            &mut rows as &mut dyn MatrixMut<f64>,
            &mut apart,
            &mut major,
            &mut slices,
        ] {
            let mut cells = nan_cells::<f64>(factor.scratch(2, regions));
            factor
                .apply_with(block, threads, MemStack::new_any(&mut cells))
                .expect("one region at least");
        }
    }
    [
        bits(&rows.concat()),
        bits(&apart.0.concat()),
        bits(&data),
        bits(&slices.concat()),
    ]
}

fn assert_spread_changes_no_bit(factor: &ApproxInverse) {
    let alone = spread_bits(factor, threads::ONE, threads::ONE);
    assert_eq!(alone[0].len(), alone[1].len());
    for layout in &alone[2..] {
        assert_eq!(layout, &alone[1], "in place as apart through the scratch");
    }
    let eight = Threads::Count(std::num::NonZeroUsize::new(8).unwrap());
    for threads in threads::BUDGETS.into_iter().chain([eight]) {
        for regions in [threads::ONE, threads::BUDGETS[2], threads] {
            assert_eq!(
                spread_bits(factor, threads, regions),
                alone,
                "{threads:?} on {regions:?}'s regions"
            );
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
