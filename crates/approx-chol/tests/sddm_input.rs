#[path = "common/laplacian_prop.rs"]
mod laplacian_prop;

use approx_chol::{CsrRef, Grounded, GroundedError, Laplacian, LaplacianError, Sddm};
use laplacian_prop::{laplacian_csr_strategy, sddm_csr_strategy};
use proptest::prelude::*;

const MAX: f64 = f64::MAX;
const FLOOR: f64 = f64::MIN_POSITIVE / f64::EPSILON;

fn path(weights: &[f64]) -> Laplacian {
    let n = weights.len() as u32 + 1;
    let row_ptrs = (0..=n).map(|row| row.min(n - 1)).collect();
    Laplacian::new(row_ptrs, (1..n).collect(), weights.to_vec()).expect("a path Laplacian")
}

fn isolated(n: u32) -> Laplacian {
    Laplacian::new(vec![0; n as usize + 1], vec![], vec![]).expect("an edgeless Laplacian")
}

/// Both typed constructors that take surplus must reject identically.
fn rejections(laplacian: Laplacian, surplus: Vec<f64>) -> [GroundedError; 2] {
    [
        Grounded::new(laplacian.clone(), surplus.clone()).expect_err("Grounded::new"),
        Sddm::with_surplus(laplacian, surplus).expect_err("Sddm::with_surplus"),
    ]
}

#[test]
fn each_sum_check_rejects_through_its_owner() {
    assert_eq!(
        Laplacian::new(vec![0, 1, 2, 2], vec![1, 2], vec![MAX, MAX]),
        Err(LaplacianError::DegreeOverflow { vertex: 1 }),
        "degree"
    );
    let cases = [
        (
            "degree plus surplus",
            path(&[MAX / 2.0]),
            vec![0.0, MAX],
            GroundedError::DiagonalOverflow { vertex: 1 },
        ),
        (
            "degree plus surplus below the floor",
            isolated(2),
            vec![1.0, FLOOR / 2.0],
            GroundedError::DiagonalTooSmall { vertex: 1 },
        ),
        (
            "surplus total",
            isolated(2),
            vec![MAX, MAX],
            GroundedError::SurplusOverflow,
        ),
    ];
    for (label, laplacian, surplus, expected) in cases {
        assert_eq!(
            rejections(laplacian, surplus),
            [expected.clone(), expected],
            "{label}"
        );
    }
}

#[test]
fn laplacian_rejects_each_entry_at_its_edge() {
    let cases = [
        (
            "below the diagonal",
            vec![0, 1, 1],
            vec![0],
            vec![1.0],
            LaplacianError::NotStrictlyUpper { edge: (0, 0) },
        ),
        (
            "unsorted",
            vec![0, 2, 2, 2],
            vec![2, 1],
            vec![1.0, 1.0],
            LaplacianError::UnsortedNeighbors { row: 0 },
        ),
        (
            "negative weight",
            vec![0, 1, 1],
            vec![1],
            vec![-1.0],
            LaplacianError::InvalidWeight { edge: (0, 1) },
        ),
        (
            "NaN weight",
            vec![0, 1, 1],
            vec![1],
            vec![f64::NAN],
            LaplacianError::InvalidWeight { edge: (0, 1) },
        ),
        (
            "weight below the floor",
            vec![0, 1, 1],
            vec![1],
            vec![FLOOR / 2.0],
            LaplacianError::WeightTooSmall { edge: (0, 1) },
        ),
    ];
    for (label, row_ptrs, neighbors, weights, expected) in cases {
        assert_eq!(
            Laplacian::new(row_ptrs, neighbors, weights),
            Err(expected),
            "{label}"
        );
    }
}

#[test]
fn surplus_rejects_each_entry_at_its_vertex() {
    let subnormal = f64::MIN_POSITIVE / 2.0;
    for (label, surplus, expected) in [
        (
            "short",
            vec![1.0],
            GroundedError::LengthMismatch {
                expected: 2,
                got: 1,
            },
        ),
        (
            "negative",
            vec![1.0, -1.0],
            GroundedError::InvalidSurplus { vertex: 1 },
        ),
        (
            "infinite",
            vec![f64::INFINITY, 1.0],
            GroundedError::InvalidSurplus { vertex: 0 },
        ),
        (
            "subnormal",
            vec![1.0, subnormal],
            GroundedError::InvalidSurplus { vertex: 1 },
        ),
    ] {
        assert_eq!(
            rejections(path(&[1.0]), surplus),
            [expected.clone(), expected],
            "{label}"
        );
    }
}

#[test]
fn surplus_picks_the_variant() {
    let laplacian = path(&[1.0, 2.0]);
    assert_eq!(
        Sddm::with_surplus(laplacian.clone(), vec![0.0; 3]),
        Ok(Sddm::Laplacian(laplacian.clone()))
    );
    assert_eq!(
        Grounded::new(laplacian.clone(), vec![0.0; 3]),
        Err(GroundedError::NoSurplus)
    );
    let grounded = Grounded::new(laplacian.clone(), vec![0.0, 0.5, 0.0]).expect("grounded");
    assert_eq!(grounded.surplus(), &[0.0, 0.5, 0.0]);
    assert_eq!(
        Sddm::with_surplus(laplacian, vec![0.0, 0.5, 0.0]),
        Ok(Sddm::Grounded(grounded))
    );
}

#[test]
fn csr_converts_to_its_upper_rows_and_surplus() {
    let row_ptrs = [0u32, 2, 5, 7];
    let col_indices = [0u32, 1, 0, 1, 2, 1, 2];
    let values = [1.0, -1.0, -1.0, 3.0, -2.0, -2.0, 2.0];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 3).expect("valid CSR");

    let Sddm::Laplacian(laplacian) = Sddm::try_from(csr).expect("a Laplacian") else {
        panic!("balanced rows convert to a Laplacian");
    };
    assert_eq!(laplacian.n(), 3);
    assert_eq!(laplacian.row_ptrs(), &[0, 1, 2, 2]);
    assert_eq!(laplacian.neighbors(), &[1, 2]);
    assert_eq!(laplacian.weights(), &[1.0, 2.0]);

    let values = [1.5, -1.0, -1.0, 3.0, -2.0, -2.0, 2.0];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 3).expect("valid CSR");
    let Sddm::Grounded(grounded) = Sddm::try_from(csr).expect("grounded") else {
        panic!("a row with surplus converts to a grounded SDDM");
    };
    assert_eq!(grounded.laplacian(), &laplacian);
    assert_eq!(grounded.surplus(), &[0.5, 0.0, 0.0]);
}

/// Rebuilt from its own parts, a converted input passes the typed constructors unchanged.
fn revalidates(row_ptrs: &[u32], col_indices: &[u32], values: &[f64], n: u32) {
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    let sddm = Sddm::try_from(csr).expect("SDDM");
    let laplacian = sddm.laplacian();
    let rebuilt = Laplacian::new(
        laplacian.row_ptrs().to_vec(),
        laplacian.neighbors().to_vec(),
        laplacian.weights().to_vec(),
    );
    assert_eq!(rebuilt.as_ref(), Ok(laplacian));
    if let Sddm::Grounded(grounded) = &sddm {
        let regrounded = Grounded::new(rebuilt.expect("checked"), grounded.surplus().to_vec());
        assert_eq!(regrounded.as_ref(), Ok(grounded));
    }
}

proptest! {
    #[test]
    fn converted_laplacians_revalidate((rp, ci, vals, n) in laplacian_csr_strategy()) {
        revalidates(&rp, &ci, &vals, n);
    }

    #[test]
    fn converted_sddms_revalidate((rp, ci, vals, n) in sddm_csr_strategy()) {
        revalidates(&rp, &ci, &vals, n);
    }
}
