use approx_chol_next::input::{
    BipartiteGram, Csr, CsrError, FactorWidth1, FactorWidth2, Laplacian, SignedGraph, UpperCsr,
    Weights,
};
use approx_chol_next::{factorize, Config, FromCsrError, ReduceError, ScalingPolicy, Sgdd};
use num_traits::PrimInt;

use super::{assert_names_a_defect, grounded, reduce, reduce_upper};

type Rejected<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64], &'a [FromCsrError]);
type Admitted<'a> = (&'a str, &'a [u32], &'a [u32], &'a [f64]);

fn csr(error: CsrError) -> FromCsrError {
    FromCsrError::Csr(error)
}

fn out_of_range(vertex: usize) -> FromCsrError {
    FromCsrError::Reduce(ReduceError::OutOfRange { vertex })
}

/// The smallest nonzero magnitude admitted (#163).
fn floor() -> f64 {
    f64::MIN_POSITIVE / f64::EPSILON
}

/// The same matrix with every row's entries reversed, which forces the canonicalizing copy.
fn reversed_rows<T: Copy>(row_ptrs: &[u32], columns: &[u32], values: &[T]) -> (Vec<u32>, Vec<T>) {
    let (mut reversed_columns, mut reversed_values) = (Vec::new(), Vec::new());
    for span in row_ptrs.windows(2) {
        let span = span[0] as usize..span[1] as usize;
        reversed_columns.extend(columns[span.clone()].iter().rev());
        reversed_values.extend(values[span].iter().rev());
    }
    (reversed_columns, reversed_values)
}

#[test]
fn out_of_class_input_is_rejected_at_its_reported_position() {
    let (max, quarter, tiny) = (f64::MAX, f64::MAX / 4.0, 1e-310);
    let over_eight_ulps = 1.0 + 9.0 * f64::EPSILON;
    let cases: [Rejected<'_>; 18] = [
        (
            "missing transpose",
            &[0, 2, 3],
            &[0, 1, 1],
            &[1.0, -1.0, 1.0],
            &[
                csr(CsrError::NotSymmetric { row: 0 }),
                csr(CsrError::NotSymmetric { row: 1 }),
            ],
        ),
        // The lower entry is stored and its upper is absent: the mirror cursor skips past it.
        (
            "missing upper mirror",
            &[0, 1, 3],
            &[0, 0, 1],
            &[1.0, -1.0, 1.0],
            &[
                csr(CsrError::NotSymmetric { row: 0 }),
                csr(CsrError::NotSymmetric { row: 1 }),
            ],
        ),
        (
            "unequal transpose",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -1.0, -2.0, 2.0],
            &[
                csr(CsrError::NotSymmetric { row: 0 }),
                csr(CsrError::NotSymmetric { row: 1 }),
            ],
        ),
        (
            "mirrors nine ulps apart",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0, -1.0, -over_eight_ulps, 2.0],
            &[
                csr(CsrError::NotSymmetric { row: 0 }),
                csr(CsrError::NotSymmetric { row: 1 }),
            ],
        ),
        (
            "stored NaN",
            &[0, 1],
            &[0],
            &[f64::NAN],
            &[csr(CsrError::OutOfRange { row: 0 })],
        ),
        (
            "stored infinity off the diagonal",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0],
            &[
                csr(CsrError::OutOfRange { row: 0 }),
                csr(CsrError::OutOfRange { row: 1 }),
            ],
        ),
        // Descending columns in row 0, so the row must be the caller's, not the canonical copy's.
        (
            "stored NaN in a later row of non-canonical input",
            &[0, 2, 5, 7],
            &[1, 0, 2, 1, 0, 2, 1],
            &[-1.0, 1.0, -1.0, 2.0, f64::NAN, 1.0, -1.0],
            &[csr(CsrError::OutOfRange { row: 1 })],
        ),
        (
            "overflow via coalesced duplicates",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[max, -max, -max, -max, -max, max],
            &[
                csr(CsrError::OutOfRange { row: 0 }),
                csr(CsrError::OutOfRange { row: 1 }),
            ],
        ),
        (
            "overflow via duplicate diagonal",
            &[0, 2],
            &[0, 0],
            &[max, max],
            &[csr(CsrError::OutOfRange { row: 0 })],
        ),
        // Every entry in range and the matrix GDD, but the centre's `Σ|a_ij|` passes `MAX/4`.
        (
            "absolute row sum past a quarter of the maximum",
            &[0, 3, 5, 7],
            &[0, 1, 2, 0, 1, 0, 2],
            &[
                quarter,
                -0.6 * quarter,
                -0.6 * quarter,
                -0.6 * quarter,
                quarter,
                -0.6 * quarter,
                quarter,
            ],
            &[out_of_range(0)],
        ),
        (
            "diagonal past a quarter of the maximum",
            &[0, 1],
            &[0],
            &[max],
            &[out_of_range(0)],
        ),
        // Normal diagonals; the edge's pivot alone would have no finite inverse (#163).
        (
            "subnormal edge",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -tiny, -tiny, 1.0],
            &[
                csr(CsrError::OutOfRange { row: 0 }),
                csr(CsrError::OutOfRange { row: 1 }),
            ],
        ),
        (
            "subnormal diagonal",
            &[0, 1],
            &[0],
            &[tiny],
            &[csr(CsrError::OutOfRange { row: 0 })],
        ),
        (
            "normal edge below MIN_POSITIVE/EPSILON",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -floor() / 2.0, -floor() / 2.0, 1.0],
            &[
                csr(CsrError::OutOfRange { row: 0 }),
                csr(CsrError::OutOfRange { row: 1 }),
            ],
        ),
        (
            "subnormal edge in a later row",
            &[0, 1, 3, 5],
            &[0, 1, 2, 1, 2],
            &[1.0, 1.0, -tiny, -tiny, 1.0],
            &[
                csr(CsrError::OutOfRange { row: 1 }),
                csr(CsrError::OutOfRange { row: 2 }),
            ],
        ),
        (
            "negative diagonal",
            &[0, 1],
            &[0],
            &[-1.0],
            &[FromCsrError::Reduce(ReduceError::NonpositiveDiagonal {
                vertex: 0,
            })],
        ),
        (
            "zero diagonal beside an edge",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[0.0, -1.0, -1.0, 1.0],
            &[FromCsrError::Reduce(ReduceError::NonpositiveDiagonal {
                vertex: 0,
            })],
        ),
        // Components {0, 2} and {1, 3}: the vertex is in the caller's numbering, not the block's.
        (
            "nonpositive diagonal in an interleaved later block",
            &[0, 2, 4, 6, 8],
            &[0, 2, 1, 3, 0, 2, 1, 3],
            &[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
            &[FromCsrError::Reduce(ReduceError::NonpositiveDiagonal {
                vertex: 3,
            })],
        ),
    ];

    for (label, row_ptrs, columns, values, defects) in cases {
        let observed = reduce(row_ptrs, columns, values).expect_err(label);
        assert_names_a_defect(label, observed, defects);
    }
}

#[test]
fn out_of_class_upper_storage_is_rejected_at_its_reported_position() {
    let cases: [Rejected<'_>; 3] = [
        (
            "entry below the diagonal",
            &[0, 1, 3],
            &[0, 0, 1],
            &[1.0, -1.0, 1.0],
            &[csr(CsrError::BelowDiagonal { row: 1 })],
        ),
        (
            "entry below the diagonal in non-canonical input",
            &[0, 2, 4],
            &[1, 0, 1, 0],
            &[-1.0, 1.0, 1.0, -1.0],
            &[csr(CsrError::BelowDiagonal { row: 1 })],
        ),
        (
            "stored NaN",
            &[0, 2, 3],
            &[0, 1, 1],
            &[1.0, f64::NAN, 1.0],
            &[csr(CsrError::OutOfRange { row: 0 })],
        ),
    ];
    for (label, row_ptrs, columns, values, defects) in cases {
        let observed = reduce_upper(row_ptrs, columns, values).expect_err(label);
        assert_names_a_defect(label, observed, defects);
    }
}

/// Shapes a stricter check would reject, admitted as given and through the canonicalizing copy.
#[test]
fn in_class_input_is_admitted_on_both_paths() {
    let next_after_one = f64::from_bits(1.0f64.to_bits() + 1);
    let eight_ulps = 1.0 + 8.0 * f64::EPSILON;
    let quarter = f64::MAX / 4.0;
    let cases: [Admitted<'_>; 8] = [
        (
            "one-ulp transpose difference",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0, -1.0, -next_after_one, 2.0],
        ),
        (
            "eight-ulp transpose difference",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0, -1.0, -eight_ulps, 2.0],
        ),
        (
            "transposes equal only after coalescing",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[2.0, -0.25, -0.75, -0.5, -0.5, 2.0],
        ),
        (
            "stored zero without a mirror",
            &[0, 2, 3],
            &[0, 1, 1],
            &[1.0, 0.0, 1.0],
        ),
        (
            "block-diagonal SDDM",
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
        ),
        // Generalized dominance: a signature folds it into a Z-matrix.
        (
            "positive off-diagonal",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[5.0, 1.0, 1.0, 4.0],
        ),
        (
            "diagonal at a quarter of the maximum",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[quarter, -quarter / 2.0, -quarter / 2.0, quarter],
        ),
        (
            "edge at MIN_POSITIVE/EPSILON",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -floor(), -floor(), 1.0],
        ),
    ];

    for (label, row_ptrs, columns, values) in cases {
        let (reversed_columns, reversed_values) = reversed_rows(row_ptrs, columns, values);
        let given = reduce(row_ptrs, columns, values).unwrap_or_else(|e| panic!("{label}: {e:?}"));
        let copied = reduce(row_ptrs, &reversed_columns, &reversed_values)
            .unwrap_or_else(|e| panic!("{label}, reversed: {e:?}"));
        assert_eq!(format!("{given:?}"), format!("{copied:?}"), "{label}");
    }
}

#[test]
#[ignore = "every entry point panics dividing by zero vertices in Partitioned::extract"]
fn an_empty_system_is_admitted_by_every_entry_point() {
    let policy = ScalingPolicy::default();
    let csr = Csr::<f64>::new(0, &[0], &[], &[]).unwrap();
    let upper = UpperCsr::<f64>::new(0, &[0], &[], &[]).unwrap();
    let laplacian = Laplacian::<f64>::new(0, &[], &[], Weights::unit()).unwrap();
    let signed = SignedGraph::<f64>::new(0, &[], &[], &[], Weights::unit()).unwrap();
    let gram = FactorWidth2::<f64>::new(0, &[], &[], &[], &[]).unwrap();
    let side = FactorWidth1::<f64>::new(0, &[], None).unwrap();
    let bipartite = BipartiteGram::new(side, side, Weights::unit()).unwrap();
    let reduced: [Sgdd; 6] = [
        Sgdd::from_csr(csr, policy).expect("CSR"),
        Sgdd::from_upper_csr(upper, policy).expect("upper CSR"),
        Sgdd::try_from(laplacian).expect("Laplacian"),
        Sgdd::try_from(signed).expect("signed graph"),
        Sgdd::from_gram(gram, policy).expect("Gram"),
        Sgdd::from_bipartite_gram(bipartite, policy).expect("bipartite Gram"),
    ];
    for sgdd in reduced {
        assert_eq!((sgdd.n(), sgdd.blocks().len()), (0, 0));
    }
}

#[test]
fn singleton_systems_are_admitted() {
    let zero = reduce(&[0, 1], &[0], &[0.0]).expect("zero singleton");
    assert_eq!(
        (zero.n(), zero.blocks().len()),
        (1, 0),
        "a zero row is no block"
    );

    let positive = reduce(&[0, 1], &[0], &[2.0]).expect("positive singleton");
    assert!(grounded(&positive));
}

#[test]
fn a_diagonal_at_a_quarter_of_the_maximum_still_solves() {
    let quarter = f64::MAX / 4.0;
    let sgdd = reduce(&[0, 1], &[0], &[quarter]).expect("diagonal at the bound");
    let solution = factorize(&sgdd, &Config::default())
        .solve(&[quarter])
        .expect("solve");
    assert!((solution[0] - 1.0).abs() < 1e-12, "{solution:?}");
}

/// The #163 range in `f32`'s own bounds: `MIN_POSITIVE/EPSILON` is `2^-103`, not `f64`'s `2^-970`.
#[test]
fn f32_range_is_judged_in_f32() {
    let (floor, quarter) = (f32::MIN_POSITIVE / f32::EPSILON, f32::MAX / 4.0);
    let pair = |diagonal: f32, edge: f32| {
        reduce(&[0, 2, 4], &[0, 1, 0, 1], &[diagonal, edge, edge, diagonal])
    };
    let rows = [
        csr(CsrError::OutOfRange { row: 0 }),
        csr(CsrError::OutOfRange { row: 1 }),
    ];
    for (label, diagonal, edge, defects) in [
        ("subnormal edge", 1.0, -1e-40, &rows[..]),
        ("normal edge below the floor", 1.0, -floor / 2.0, &rows[..]),
        (
            "diagonal past a quarter of the maximum",
            f32::MAX,
            -1.0,
            &[out_of_range(0), out_of_range(1)][..],
        ),
    ] {
        let observed = pair(diagonal, edge).expect_err(label);
        assert_names_a_defect(label, observed, defects);
    }
    for (label, diagonal, edge) in [
        ("edge at the floor", 1.0, -floor),
        (
            "diagonal at a quarter of the maximum",
            quarter,
            -quarter / 2.0,
        ),
    ] {
        pair(diagonal, edge).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }
}

fn grid_laplacian(rows: usize, cols: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let (mut row_ptrs, mut columns, mut values) = (vec![0u32], Vec::new(), Vec::new());
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            let mut neighbours = Vec::new();
            if r > 0 {
                neighbours.push(v - cols);
            }
            if c > 0 {
                neighbours.push(v - 1);
            }
            let degree = neighbours.len() + usize::from(c + 1 < cols) + usize::from(r + 1 < rows);
            neighbours.push(v);
            if c + 1 < cols {
                neighbours.push(v + 1);
            }
            if r + 1 < rows {
                neighbours.push(v + cols);
            }
            for j in neighbours {
                columns.push(j as u32);
                values.push(if j == v { degree as f64 } else { -1.0 });
            }
            row_ptrs.push(columns.len() as u32);
        }
    }
    (row_ptrs, columns, values)
}

fn widen<I: PrimInt>(indices: &[u32]) -> Vec<I> {
    (indices.iter())
        .map(|&index| I::from(index).expect("fits"))
        .collect()
}

fn bits(sgdd: &Sgdd, rhs: &[f64]) -> (String, Vec<u64>) {
    let solution = factorize(sgdd, &Config::default())
        .solve(rhs)
        .expect("solve");
    (
        format!("{sgdd:?}"),
        solution.iter().map(|x| x.to_bits()).collect(),
    )
}

/// Descending columns force the canonicalizing copy; the `u64` copy is rewritten in its own index type.
#[test]
fn canonical_and_reordered_ingestion_agree_bit_for_bit() {
    let (row_ptrs, columns, values) = grid_laplacian(6, 7);
    let n = row_ptrs.len() - 1;
    let (reversed_columns, reversed_values) = reversed_rows(&row_ptrs, &columns, &values);
    let rhs: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let policy = ScalingPolicy::default();
    let reduce = |csr| Sgdd::from_csr(csr, policy).expect("grid");

    let canonical = bits(
        &reduce(Csr::new(n, &row_ptrs, &columns, &values).unwrap()),
        &rhs,
    );
    let reordered = Csr::new(n, &row_ptrs, &reversed_columns, &reversed_values).unwrap();
    assert_eq!(canonical, bits(&reduce(reordered), &rhs));

    let (wide_row_ptrs, wide_columns) = (widen::<u64>(&row_ptrs), widen::<u64>(&reversed_columns));
    let wide = Csr::new(n, &wide_row_ptrs, &wide_columns, &reversed_values).unwrap();
    let wide = Sgdd::from_csr(wide, policy).expect("grid");
    assert_eq!(canonical, bits(&wide, &rhs));

    let upper = |row_ptrs: &[u32], columns: &[u32], values: &[f64]| {
        let (mut ptrs, mut kept_columns, mut kept_values) = (vec![0u32], Vec::new(), Vec::new());
        for (row, span) in row_ptrs.windows(2).enumerate() {
            for k in span[0] as usize..span[1] as usize {
                if columns[k] as usize >= row {
                    kept_columns.push(columns[k]);
                    kept_values.push(values[k]);
                }
            }
            ptrs.push(kept_columns.len() as u32);
        }
        (ptrs, kept_columns, kept_values)
    };
    for (label, columns, values) in [
        ("canonical", &columns, &values),
        ("reversed", &reversed_columns, &reversed_values),
    ] {
        let (ptrs, kept_columns, kept_values) = upper(&row_ptrs, columns, values);
        let triangle = UpperCsr::new(n, &ptrs, &kept_columns, &kept_values).unwrap();
        let triangle = Sgdd::from_upper_csr(triangle, policy).expect("grid");
        assert_eq!(canonical, bits(&triangle, &rhs), "{label} upper storage");
    }
}

// Splitting the diagonal matters: a coalescing bug there is invisible to the sign and symmetry checks.
#[test]
fn unsorted_and_split_entries_reduce_bit_identically() {
    let canonical = reduce(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    )
    .expect("path");
    let shuffled = reduce(
        &[0, 4, 7, 9],
        &[1, 0, 1, 0, 2, 0, 1, 2, 1],
        &[-0.5, 0.25, -0.5, 0.75, -1.0, -1.0, 2.0, 1.0, -1.0],
    )
    .expect("split path");
    let rhs = [1.0, 0.0, -1.0];
    assert_eq!(bits(&canonical, &rhs), bits(&shuffled, &rhs));
}
