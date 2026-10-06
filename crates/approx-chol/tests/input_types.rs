use approx_chol::{CsrError, Grounded, GroundedError, Laplacian, LaplacianError, Sddm};

fn path() -> Laplacian {
    Laplacian::new(vec![0, 1, 2, 2], vec![1, 2], vec![1.0, 1.0]).expect("valid path")
}

#[test]
fn every_laplacian_error_is_reachable() {
    let max = f64::MAX;
    #[allow(clippy::type_complexity)]
    let cases: [(&str, Vec<u32>, Vec<u32>, Vec<f64>, LaplacianError); 8] = [
        (
            "no row pointers",
            vec![],
            vec![],
            vec![],
            LaplacianError::Structure(CsrError::RowPtrsLenMismatch {
                expected: 1,
                got: 0,
            }),
        ),
        (
            "neighbor past the last vertex",
            vec![0, 1, 1],
            vec![2],
            vec![1.0],
            LaplacianError::Structure(CsrError::ColumnIndexOutOfBounds {
                position: 0,
                col: 2,
                n: 2,
            }),
        ),
        (
            "a neighbor below its row",
            vec![0, 0, 1],
            vec![0],
            vec![1.0],
            LaplacianError::NotStrictlyUpper { edge: (1, 0) },
        ),
        (
            "a self loop",
            vec![0, 1, 1],
            vec![0],
            vec![1.0],
            LaplacianError::NotStrictlyUpper { edge: (0, 0) },
        ),
        (
            "neighbors out of order",
            vec![0, 2, 2, 2],
            vec![2, 1],
            vec![1.0, 1.0],
            LaplacianError::UnsortedNeighbors { row: 0 },
        ),
        (
            "a zero weight",
            vec![0, 1, 1],
            vec![1],
            vec![0.0],
            LaplacianError::InvalidWeight { edge: (0, 1) },
        ),
        (
            "a NaN weight",
            vec![0, 1, 1],
            vec![1],
            vec![f64::NAN],
            LaplacianError::InvalidWeight { edge: (0, 1) },
        ),
        // Each weight is finite; vertex 1 sums both.
        (
            "a degree that overflows",
            vec![0, 1, 2, 2],
            vec![1, 2],
            vec![max, max],
            LaplacianError::DegreeOverflow { vertex: 1 },
        ),
    ];
    for (label, row_ptrs, neighbors, weights, expected) in cases {
        let error = Laplacian::new(row_ptrs, neighbors, weights).expect_err(label);
        assert_eq!(error, expected, "{label}");
    }
}

#[test]
fn every_grounded_error_is_reachable() {
    let max = f64::MAX;
    let big = || Laplacian::new(vec![0, 1, 1], vec![1], vec![max]).expect("valid edge");
    let cases = [
        (
            "one surplus short",
            path(),
            vec![1.0, 0.0],
            GroundedError::LengthMismatch {
                expected: 3,
                got: 2,
            },
        ),
        (
            "a negative surplus",
            path(),
            vec![1.0, -1.0, 0.0],
            GroundedError::InvalidSurplus { vertex: 1 },
        ),
        (
            "an infinite surplus",
            path(),
            vec![f64::INFINITY, 0.0, 0.0],
            GroundedError::InvalidSurplus { vertex: 0 },
        ),
        ("all zero", path(), vec![0.0; 3], GroundedError::NoSurplus),
        (
            "a total that overflows",
            path(),
            vec![max, 0.0, max],
            GroundedError::SurplusOverflow,
        ),
        // Finite degree and finite surplus, but not their sum.
        (
            "a diagonal that overflows",
            big(),
            vec![0.0, max],
            GroundedError::DiagonalOverflow { vertex: 1 },
        ),
    ];
    for (label, laplacian, surplus, expected) in cases {
        let error = Grounded::new(laplacian, surplus).expect_err(label);
        assert_eq!(error, expected, "{label}");
    }
}

/// The one place a caller with surplus in hand need not know whether any is positive.
#[test]
fn with_surplus_is_a_laplacian_exactly_when_every_surplus_is_zero() {
    assert!(matches!(
        Sddm::with_surplus(path(), vec![0.0; 3]),
        Ok(Sddm::Laplacian(_))
    ));
    assert!(matches!(
        Sddm::with_surplus(path(), vec![0.0, 0.5, 0.0]),
        Ok(Sddm::Grounded(_))
    ));
    assert_eq!(
        Sddm::with_surplus(path(), vec![0.0; 2]).expect_err("short surplus"),
        GroundedError::LengthMismatch {
            expected: 3,
            got: 2
        }
    );
}
