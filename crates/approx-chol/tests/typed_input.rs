use approx_chol::{
    factorize, AdjacencyError, CsrRef, Laplacian, LaplacianError, Sddm, SddmError, SurplusDefect,
    WeightDefect,
};

type Arrays = (Vec<u32>, Vec<u32>, Vec<f64>);

/// The path 0-1-2-3 with weights 1, 2, 1.
fn path() -> Arrays {
    (vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1.0, 2.0, 1.0])
}

/// The same edges as a symmetric CSR, each diagonal its degree plus `surplus`.
fn path_csr(surplus: [f64; 4]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let degree = [1.0, 3.0, 3.0, 1.0];
    let values = vec![
        degree[0] + surplus[0],
        -1.0,
        -1.0,
        degree[1] + surplus[1],
        -2.0,
        -2.0,
        degree[2] + surplus[2],
        -1.0,
        -1.0,
        degree[3] + surplus[3],
    ];
    (
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        values,
    )
}

fn solve(sddm: Sddm, b: &[f64]) -> Vec<f64> {
    factorize(sddm).solve(b).expect("solve")
}

fn from_csr(surplus: [f64; 4]) -> Sddm {
    let (rp, ci, v) = path_csr(surplus);
    Sddm::try_from(CsrRef::new(&rp, &ci, &v, 4).expect("valid CSR")).expect("an SDDM")
}

#[test]
fn a_typed_laplacian_factors_as_its_csr_does() {
    let (rp, nb, w) = path();
    let b = [1.0, -1.0, 2.0, -2.0];
    let typed = solve(Laplacian::new(rp, nb, w).expect("a Laplacian").into(), &b);
    assert_eq!(typed, solve(from_csr([0.0; 4]), &b));
}

#[test]
fn a_typed_sddm_factors_as_its_csr_does() {
    let (rp, nb, w) = path();
    let surplus = [0.5, 0.0, 0.0, 2.0];
    let b = [1.0, -1.0, 2.0, -2.0];
    let typed = solve(Sddm::new(rp, nb, w, surplus.to_vec()).expect("an SDDM"), &b);
    assert_eq!(typed, solve(from_csr(surplus), &b));
}

/// No vertex holds surplus, so the system floats and the solution is the zero-mean one.
#[test]
fn an_all_zero_surplus_floats() {
    let (rp, nb, w) = path();
    let x = solve(
        Sddm::new(rp, nb, w, vec![0.0; 4]).expect("an SDDM"),
        &[1.0, -1.0, 1.0, -1.0],
    );
    assert!(x.iter().sum::<f64>().abs() < 1e-12, "{x:?}");
}

#[test]
fn an_empty_laplacian_factors_and_solves() {
    let laplacian = Laplacian::<f64>::new(vec![0], vec![], vec![]).expect("no vertices");
    assert_eq!(laplacian.n(), 0);
    assert_eq!(
        factorize(laplacian).solve(&[]).expect("solve"),
        Vec::<f64>::new()
    );
}

#[test]
fn a_rejected_adjacency_names_its_defect() {
    let tiny = f64::MIN_POSITIVE;
    let cases: [(&str, Arrays, AdjacencyError); 12] = [
        (
            "empty row_ptrs",
            (vec![], vec![], vec![]),
            AdjacencyError::RowPtrsEmpty,
        ),
        (
            "row_ptrs from 1",
            (vec![1, 1], vec![], vec![]),
            AdjacencyError::RowPtrsMustStartAtZero { got: 1 },
        ),
        (
            "fewer weights",
            (vec![0, 1, 1], vec![1], vec![]),
            AdjacencyError::NeighborsWeightsLenMismatch {
                neighbors: 1,
                weights: 0,
            },
        ),
        (
            "end short of neighbors",
            (vec![0, 0, 0], vec![1], vec![1.0]),
            AdjacencyError::RowPtrsEndMismatch { end: 0, len: 1 },
        ),
        (
            "decreasing row_ptrs",
            (vec![0, 2, 1, 2], vec![1, 2], vec![1.0, 1.0]),
            AdjacencyError::RowPtrsDecrease { row: 1 },
        ),
        (
            "a self loop",
            (vec![0, 1, 1], vec![0], vec![1.0]),
            AdjacencyError::NotStrictlyUpper { edge: (0, 0) },
        ),
        (
            "a lower neighbor",
            (vec![0, 0, 1], vec![0], vec![1.0]),
            AdjacencyError::NotStrictlyUpper { edge: (1, 0) },
        ),
        (
            "a repeated neighbor",
            (vec![0, 2, 2, 2], vec![1, 1], vec![1.0, 1.0]),
            AdjacencyError::Unsorted { edge: (0, 1) },
        ),
        (
            "a neighbor past n",
            (vec![0, 1, 1], vec![2], vec![1.0]),
            AdjacencyError::NeighborOutOfBounds { edge: (0, 2), n: 2 },
        ),
        (
            "a NaN weight",
            (vec![0, 1, 1], vec![1], vec![f64::NAN]),
            AdjacencyError::Weight {
                edge: (0, 1),
                defect: WeightDefect::NonFinite,
            },
        ),
        (
            "a negative weight",
            (vec![0, 1, 1], vec![1], vec![-1.0]),
            AdjacencyError::Weight {
                edge: (0, 1),
                defect: WeightDefect::NotPositive,
            },
        ),
        (
            "a weight below the floor",
            (vec![0, 1, 1], vec![1], vec![tiny]),
            AdjacencyError::Weight {
                edge: (0, 1),
                defect: WeightDefect::BelowFloor,
            },
        ),
    ];
    for (label, (rp, nb, w), expected) in cases {
        assert_eq!(
            Laplacian::new(rp.clone(), nb.clone(), w.clone()).expect_err(label),
            LaplacianError::Adjacency(expected.clone()),
            "{label}"
        );
        assert_eq!(
            Sddm::new(rp.clone(), nb, w, vec![0.0; rp.len().saturating_sub(1)]).expect_err(label),
            SddmError::Adjacency(expected),
            "{label}"
        );
    }
}

#[test]
fn a_degree_that_overflows_is_rejected() {
    let max = f64::MAX;
    assert_eq!(
        Laplacian::new(vec![0, 2, 2, 2], vec![1, 2], vec![max, max]).expect_err("overflow"),
        LaplacianError::DegreeNotFinite { vertex: 0 }
    );
}

#[test]
fn a_rejected_surplus_names_its_defect() {
    let max = f64::MAX;
    let edge = || (vec![0u32, 1, 1], vec![1u32], vec![1.0]);
    let cases: [(&str, Vec<f64>, SddmError); 6] = [
        (
            "one entry short",
            vec![0.0],
            SddmError::SurplusLength { len: 1, n: 2 },
        ),
        (
            "negative",
            vec![0.0, -1.0],
            SddmError::Surplus {
                vertex: 1,
                defect: SurplusDefect::Negative,
            },
        ),
        (
            "below the floor",
            vec![f64::MIN_POSITIVE, 0.0],
            SddmError::Surplus {
                vertex: 0,
                defect: SurplusDefect::BelowFloor,
            },
        ),
        (
            "NaN",
            vec![f64::NAN, 0.0],
            SddmError::DiagonalNotFinite { vertex: 0 },
        ),
        (
            "infinite",
            vec![0.0, f64::INFINITY],
            SddmError::DiagonalNotFinite { vertex: 1 },
        ),
        (
            "a ground that overflows",
            vec![max, max],
            SddmError::GroundOverflow { vertex: 0 },
        ),
    ];
    for (label, surplus, expected) in cases {
        let (rp, nb, w) = edge();
        assert_eq!(
            Sddm::new(rp, nb, w, surplus).expect_err(label),
            expected,
            "{label}"
        );
    }
}

/// Each vertex its own component, so each surplus closes on its own finite ground.
#[test]
fn grounds_overflow_per_component_not_in_total() {
    let max = f64::MAX;
    Sddm::new(vec![0, 0, 0], vec![], vec![], vec![max, max]).expect("two finite grounds");
}
