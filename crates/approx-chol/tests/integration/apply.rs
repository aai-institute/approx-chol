use crate::common::{bits, cases, entry, graph, modes, scratch, sparse, threads};

use std::array;

use approx_chol::input::Laplacian;
use approx_chol::{
    factorize, ApplyError, ApproxInverse, MatrixMut, MemStack, Real, StackReq, Threads,
};
use bits::bits;
use cases::cases;
use entry::{by_type, cast, cast_all, Entry, Factored, Labels, Types};
use graph::{Edge, Graph};
use modes::Mode;
use scratch::nan_cells;
use sparse::Sparse;

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
) -> [(&'static str, Vec<Vec<f64>>); 2] {
    let n = factor.n();
    let columns: [Vec<T>; K] = array::from_fn(|j| cast_all(&labels.to_local(&rhs(n, j))));
    let mut rows: Vec<[T; K]> = (0..n).map(|i| array::from_fn(|j| columns[j][i])).collect();
    let mut apart = Apart(columns.to_vec());
    for block in [&mut rows as &mut dyn MatrixMut<T>, &mut apart] {
        let mut cells = nan_cells::<T>(factor.scratch(K, threads));
        let stack = MemStack::new_any(&mut cells);
        factor
            .apply_with(block, threads, stack)
            .expect("its own scratch");
    }
    let caller = |column: Vec<T>| labels.to_caller(&cast_all(&column));
    let interleaved = (0..K).map(|j| caller(rows.iter().map(|row| row[j]).collect()));
    [
        ("interleaved", interleaved.collect()),
        ("apart", apart.0.into_iter().map(caller).collect()),
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

/// A block of the wrong row count, or a stack short of the scratch, errs and leaves the block as it was.
fn assert_errors_leave_the_block<T: Real>(factor: &ApproxInverse<T>) {
    let n = factor.n();
    let row = |i: usize| [cast(i as f64), cast(-(i as f64))];
    for found in [n - 1, n + 1] {
        let mut rows: Vec<[T; 2]> = (0..found).map(row).collect();
        let before = rows.clone();
        let error = factor.apply(&mut rows, Threads::All);
        assert_eq!(error, Err(ApplyError::Rows { expected: n, found }));
        assert!(rows == before, "a block of {found} rows is left as it was");
    }
    let needed = factor.scratch(2, Threads::All);
    let mut cells = nan_cells::<T>(needed);
    let short = &mut cells[..needed.size_bytes() / size_of::<T>() - 1];
    let mut rows: Vec<[T; 2]> = (0..n).map(row).collect();
    let before = rows.clone();
    let error = factor.apply_with(&mut rows, Threads::All, MemStack::new_any(short));
    assert_eq!(error, Err(ApplyError::Scratch));
    assert!(rows == before, "a block on a short stack is left as it was");
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
}
