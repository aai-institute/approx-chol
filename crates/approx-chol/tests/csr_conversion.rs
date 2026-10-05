#[path = "common/grid.rs"]
mod grid;
#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{factorize, CsrRef, Grounded, Laplacian, NotSddm, Sddm};
use laplacian_prop::{laplacian_csr_strategy, one_grounded_component_strategy};
use num_traits::Float;
use proptest::prelude::*;

#[derive(Debug, PartialEq)]
enum Variant {
    Laplacian,
    Grounded,
}

/// `n` follows from `rp`, so no case can disagree with its own row count.
fn convert<T: Float + Send + Sync + 'static>(
    rp: &[u32],
    ci: &[u32],
    vals: &[T],
) -> Result<Sddm<T>, NotSddm> {
    let csr = CsrRef::new(rp, ci, vals, (rp.len() - 1) as u32).expect("structurally valid CSR");
    Sddm::try_from(csr)
}

fn variant<T: Float + Send + Sync + 'static>(
    rp: &[u32],
    ci: &[u32],
    vals: &[T],
) -> Result<Variant, NotSddm> {
    convert(rp, ci, vals).map(|sddm| match sddm {
        Sddm::Laplacian(_) => Variant::Laplacian,
        Sddm::Grounded(_) => Variant::Grounded,
    })
}

/// The expected error is compared whole, so a shape cannot pass by being rejected
/// elsewhere and the reported coordinate cannot drift.
#[test]
fn out_of_class_input_is_rejected_at_its_reported_position() {
    let max = f64::MAX;
    #[allow(clippy::type_complexity)]
    let cases: [(&str, &[u32], &[u32], &[f64], NotSddm); 10] = [
        // Used to fall through both the diagonal and the `val < 0` edge branch,
        // silently factorizing diag(5, 4) — a confidently wrong factor.
        (
            "positive off-diagonal",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[5.0, 1.0, 1.0, 4.0],
            NotSddm::PositiveOffDiagonal { edge: (0, 1) },
        ),
        (
            "missing transpose",
            &[0, 2, 3],
            &[0, 1, 1],
            &[1.0, -1.0, 1.0],
            NotSddm::Asymmetric { edge: (0, 1) },
        ),
        // Reaches the asymmetry the mirror cursor skips past, not the one the
        // comparison rejects: the lower entry is stored and its upper is absent.
        (
            "missing upper mirror",
            &[0, 1, 3],
            &[0, 0, 1],
            &[1.0, -1.0, 1.0],
            NotSddm::Asymmetric { edge: (0, 1) },
        ),
        (
            "unequal transpose",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -1.0, -2.0, 2.0],
            NotSddm::Asymmetric { edge: (0, 1) },
        ),
        (
            "connected but not dominant",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1.0, -3.0, -3.0, 1.0],
            NotSddm::NotDiagonallyDominant { row: 0 },
        ),
        (
            "stored NaN",
            &[0, 1],
            &[0],
            &[f64::NAN],
            NotSddm::NonFiniteValue { position: 0 },
        ),
        // Descending columns in row 0, so the position must be the caller's flat
        // index and must be reported in preference to the non-canonical shape.
        (
            "stored NaN in a later row of non-canonical input",
            &[0, 2, 5, 7],
            &[1, 0, 2, 1, 0, 2, 1],
            &[-1.0, 1.0, -1.0, 2.0, f64::NAN, 1.0, -1.0],
            NotSddm::NonFiniteValue { position: 4 },
        ),
        // The three row-sum overflows below store only finite, symmetric,
        // non-positive values; one per conversion path.
        (
            "overflow via coalesced duplicates",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[max, -max, -max, -max, -max, max],
            NotSddm::NonFiniteRow { row: 0 },
        ),
        (
            "overflow on the canonical path",
            &[0, 3, 5, 7],
            &[0, 1, 2, 0, 1, 0, 2],
            &[0.0, -max, -max, -max, max, -max, max],
            NotSddm::NonFiniteRow { row: 0 },
        ),
        (
            "overflow via duplicate diagonal",
            &[0, 2],
            &[0, 0],
            &[max, max],
            NotSddm::NonFiniteRow { row: 0 },
        ),
    ];

    for (label, rp, ci, vals, expected) in cases {
        assert_eq!(convert(rp, ci, vals).expect_err(label), expected, "{label}");
    }
}

/// Shapes a stricter check would reject, but which are valid SDDM.
#[test]
fn in_class_input_is_accepted() {
    let next_after_one = f64::from_bits(1.0f64.to_bits() + 1);
    #[allow(clippy::type_complexity)]
    let cases: [(&str, &[u32], &[u32], &[f64]); 2] = [
        (
            "one-ulp transpose difference",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0, -1.0, -next_after_one, 2.0],
        ),
        (
            "transposes equal only after coalescing",
            &[0, 3, 6],
            &[0, 1, 1, 0, 0, 1],
            &[2.0, -0.25, -0.75, -0.5, -0.5, 2.0],
        ),
    ];
    for (label, rp, ci, vals) in cases {
        convert(rp, ci, vals).unwrap_or_else(|err| panic!("{label}: {err}"));
    }
}

/// `[[a+s, -a], [-a, a+s]]`: two stored terms per row, so the noise floor is `4 * eps * a`.
fn pair<T: Float + Send + Sync + 'static>(a: T, s: T) -> Result<Variant, NotSddm> {
    variant(&[0, 2, 4], &[0, 1, 0, 1], &[a + s, -a, -a, a + s])
}

fn assert_judged<T: Float + Send + Sync + core::fmt::LowerExp + 'static>(
    cases: &[(&str, T, T, Result<Variant, NotSddm>)],
) {
    for (label, a, s, expected) in cases {
        assert_eq!(&pair(*a, *s), expected, "{label}: a {a:e}, s {s:e}");
    }
}

/// The surplus is judged against the row's own summation error, in both directions:
/// above it grounds (#85), within it is noise, below it is a deficit (#91).
#[test]
fn surplus_is_judged_against_summation_error_alone() {
    let deficit = || Err(NotSddm::NotDiagonallyDominant { row: 0 });
    assert_judged::<f64>(&[
        ("one ulp above", 1.0, f64::EPSILON, Ok(Variant::Laplacian)),
        ("one ulp below", 1.0, -f64::EPSILON, Ok(Variant::Laplacian)),
        ("mid-window dominance", 1.0, 5e-11, Ok(Variant::Grounded)),
        ("mid-window deficit", 1.0, -5e-11, deficit()),
        ("far above the floor", 1e-6, 6e-12, Ok(Variant::Grounded)),
        ("above the floor", 1e-6, 6e-15, Ok(Variant::Grounded)),
        // Was discarded by a floor 1e6 coarser than rounding.
        ("just above the floor", 1e-6, 6e-18, Ok(Variant::Grounded)),
        ("below the floor", 1e-6, 6e-22, Ok(Variant::Laplacian)),
        (
            "not representable at this scale",
            1e-6,
            1e-30,
            Ok(Variant::Laplacian),
        ),
    ]);
    assert_judged::<f32>(&[
        ("one ulp above", 1.0, f32::EPSILON, Ok(Variant::Laplacian)),
        ("one ulp below", 1.0, -f32::EPSILON, Ok(Variant::Laplacian)),
        ("mid-window dominance", 1.0, 1e-6, Ok(Variant::Grounded)),
        ("mid-window deficit", 1.0, -1e-6, deficit()),
        // #85 measured the old floor at this row scale.
        ("above the floor", 1e-3, 1.2e-9, Ok(Variant::Grounded)),
        ("below the floor", 1e-3, 2e-10, Ok(Variant::Laplacian)),
    ]);
}

/// A 4-vertex star whose centre diagonal is `offset` ULPs above its exactly-dyadic
/// off-diagonal mass, so the drift is the offset and nothing else.
fn star_at_ulp_offset(offset: u64) -> Result<Variant, NotSddm> {
    let centre = f64::from_bits(4e8f64.to_bits() + offset);
    variant(
        &[0, 4, 6, 8, 10],
        &[0, 1, 2, 3, 0, 1, 0, 2, 0, 3],
        &[centre, -1e8, -2e8, -1e8, -1e8, 1e8, -2e8, 2e8, -1e8, 1e8],
    )
}

/// The floor counts a row's stored terms: scale `8e8` over 4 terms is 11.9 ULPs of the centre.
#[test]
fn the_noise_floor_grows_with_the_row_s_term_count() {
    assert_eq!(star_at_ulp_offset(4), Ok(Variant::Laplacian));
    assert_eq!(star_at_ulp_offset(24), Ok(Variant::Grounded));
}

/// The graph symmetrizes an accepted mirror pair to one value, but classification must
/// not: charging the upper value to both rows made the tolerated difference look like the
/// lower row's own surplus, so the same matrix routed differently depending on which
/// triangle held the larger magnitude.
#[test]
fn tolerated_mirror_difference_is_not_one_row_s_surplus() {
    let off = 1.0 + 5.0 * f64::EPSILON;
    for (label, vals) in [
        ("upper holds the smaller", [1.0, -1.0, -off, off]),
        ("lower holds the smaller", [off, -off, -1.0, 1.0]),
    ] {
        assert_eq!(
            variant(&[0, 2, 4], &[0, 1, 0, 1], &vals),
            Ok(Variant::Laplacian),
            "{label}"
        );
    }
}

/// `rewrite` folds each duplicate group with its own additions, so the error allowance
/// counts stored entries rather than coalesced neighbours. Ten sub-ULP duplicates are
/// absorbed one way and accumulate the other, which invented a surplus on a row that
/// sums exactly to zero.
#[test]
fn coalescing_additions_are_inside_the_error_allowance() {
    let half = f64::EPSILON / 2.0;
    let (mut rp, mut ci, mut vals) = (vec![0u32], Vec::new(), Vec::new());
    for row in 0..2u32 {
        ci.extend(core::iter::repeat_n(row, 10));
        vals.extend(core::iter::repeat_n(half, 10));
        ci.extend([row, 1 - row]);
        vals.extend([1.0, -1.0]);
        ci.extend(core::iter::repeat_n(1 - row, 10));
        vals.extend(core::iter::repeat_n(-half, 10));
        rp.push(ci.len() as u32);
    }
    assert_eq!(variant(&rp, &ci, &vals), Ok(Variant::Laplacian));
}

/// Both ends of the row-scale range, where a floor set in absolute terms fails at one
/// end or the other: 1e12-scale dominance, a 5e-11-scale system, and a surplus real
/// against its own row scale but far below any absolute floor.
#[test]
fn genuine_surplus_at_either_scale_is_kept_and_solves() {
    #[allow(clippy::type_complexity)]
    let cases: [(&str, &[u32], &[u32], &[f64], [f64; 2], [f64; 2]); 3] = [
        (
            "1e12 scale",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1e12 + 100.0, -1e12, -1e12, 1e12 + 100.0],
            [1.0, 1.0],
            [0.01, 0.01],
        ),
        (
            "5e-11 scale",
            &[0, 1, 2],
            &[0, 1],
            &[5e-11, 5e-11],
            [1.0, 2.0],
            [1.0 / 5e-11, 2.0 / 5e-11],
        ),
        // Eigenvalue 6e-15 on [1, 1], 1e9 times the error the row's own additions carry.
        (
            "surplus far below any absolute floor",
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[1e-6 + 6e-15, -1e-6, -1e-6, 1e-6 + 6e-15],
            [1.0, 1.0],
            [1.0 / 6e-15, 1.0 / 6e-15],
        ),
    ];

    for (label, rp, ci, vals, rhs, expected) in cases {
        let solution = factorize(convert(rp, ci, vals).expect(label))
            .solve(&rhs)
            .expect("solve");
        for (got, want) in solution.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs(),
                "{label}: {solution:?} vs {expected:?}"
            );
        }
    }
}

/// `|d| + d` overflowed before the excess was subtracted, rejecting a solvable row for
/// being near the top of the range rather than for anything about its balance.
#[test]
fn a_diagonal_near_the_type_maximum_still_solves() {
    let sddm = convert(&[0, 1], &[0], &[f64::MAX]).expect("max diagonal");
    let solution = factorize(sddm).solve(&[f64::MAX]).expect("solve");
    assert!((solution[0] - 1.0).abs() < 1e-12, "{solution:?}");
}

/// Row 0 splits both its diagonal and its edge in two and row 1 is out of order; a
/// split diagonal matters because coalescing it is invisible to the sign and symmetry checks.
#[test]
fn unsorted_and_split_entries_convert_to_the_canonical_form() {
    let canonical = convert(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    );
    let shuffled = convert(
        &[0, 4, 7, 9],
        &[1, 0, 1, 0, 2, 0, 1, 2, 1],
        &[-0.5, 0.25, -0.5, 0.75, -1.0, -1.0, 2.0, 1.0, -1.0],
    );
    assert_eq!(shuffled, canonical);
}

/// Descending columns force the rewriting path, so the canonical fast path must match it bit for bit.
#[test]
fn canonical_and_reordered_input_convert_identically() {
    let grid = grid::grid_laplacian(6, 7);
    let mut reversed_columns = Vec::with_capacity(grid.col_indices.len());
    let mut reversed_values = Vec::with_capacity(grid.values.len());
    for row in 0..grid.n as usize {
        let span = grid.row_ptrs[row] as usize..grid.row_ptrs[row + 1] as usize;
        reversed_columns.extend(grid.col_indices[span.clone()].iter().rev());
        reversed_values.extend(grid.values[span].iter().rev());
    }
    let reordered = convert(&grid.row_ptrs, &reversed_columns, &reversed_values);
    assert_eq!(reordered, Ok(grid.sddm()));
}

#[test]
fn a_csr_matrix_converts_to_its_typed_input() {
    let path = Laplacian::new(vec![0, 1, 2, 2], vec![1, 2], vec![1.0, 1.0]).expect("valid path");
    let typed: Sddm = Grounded::new(path, vec![2.0, 0.0, 0.0])
        .expect("valid surplus")
        .into();
    let converted = convert(
        &[0, 2, 5, 7],
        &[0, 1, 0, 1, 2, 1, 2],
        &[3.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
    );
    assert_eq!(converted, Ok(typed));
}

/// The CSR path proves edges by canonical order, the typed constructors by checking them;
/// whatever the first accepts, the second must accept unchanged.
fn assert_revalidates(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) {
    let converted = convert(row_ptrs, col_indices, values).expect("valid SDDM");
    let laplacian = converted.laplacian();
    let rebuilt = Laplacian::new(
        laplacian.row_ptrs().to_vec(),
        laplacian.neighbors().to_vec(),
        laplacian.weights().to_vec(),
    )
    .expect("a converted Laplacian passes the typed checks");
    let surplus = match &converted {
        Sddm::Laplacian(_) => vec![0.0; laplacian.n()],
        Sddm::Grounded(grounded) => grounded.surplus().to_vec(),
    };
    let retyped =
        Sddm::with_surplus(rebuilt, surplus).expect("converted surplus passes the typed checks");
    assert_eq!(retyped, converted);
}

proptest! {
    #[test]
    fn a_converted_laplacian_revalidates((row_ptrs, col_indices, values, _) in laplacian_csr_strategy()) {
        assert_revalidates(&row_ptrs, &col_indices, &values);
    }

    #[test]
    fn a_converted_grounded_sddm_revalidates(((row_ptrs, col_indices, values, _), _) in one_grounded_component_strategy()) {
        assert_revalidates(&row_ptrs, &col_indices, &values);
    }
}
