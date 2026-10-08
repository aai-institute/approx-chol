use crate::graph::Single;
use crate::sddm::Sddm;
use crate::CsrRef;

fn sddm(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Sddm<f64> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    Sddm::try_from(csr).expect("valid SDDM")
}

/// Each component's global vertices, `None` when the input is one component.
fn blocks_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Option<Vec<Vec<u32>>> {
    let (blocks, order) = sddm(row_ptrs, col_indices, values)
        .map_components(|component| {
            let len = component.eliminated() + usize::from(!component.is_grounded());
            Ok::<_, ()>(
                (0..len)
                    .map(|local| component.global(local) as u32)
                    .collect(),
            )
        })
        .expect("no component fails");
    order.map(|_| blocks)
}

/// Three pairs on interleaved vertices: grounded at one vertex, floating, grounded at both.
const ROW_PTRS: [u32; 7] = [0, 2, 4, 6, 8, 10, 12];
const COL_INDICES: [u32; 12] = [0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5];
const VALUES: [f64; 12] = [
    2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 2.0,
];

#[test]
fn surplus_never_joins_components() {
    assert_eq!(
        blocks_of(&ROW_PTRS, &COL_INDICES, &VALUES),
        Some(vec![vec![0, 3], vec![1, 4], vec![2, 5]])
    );
}

/// An isolated vertex is its own block, which makes ordering "by lowest member" observable.
#[test]
fn blocks_are_ordered_by_their_lowest_vertex() {
    let blocks = blocks_of(
        &[0, 2, 3, 4, 6],
        &[0, 3, 1, 2, 0, 3],
        &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 3], vec![1], vec![2]]));
}

#[test]
fn an_empty_input_has_no_components() {
    assert_eq!(blocks_of(&[0], &[], &[]), Some(vec![]));
}

/// Each grounded component gets its own ground slot, joined to its own surplus vertices only.
#[test]
fn each_grounded_component_appends_its_own_ground_slot() {
    let (built, _) = sddm(&ROW_PTRS, &COL_INDICES, &VALUES)
        .map_components(|component| {
            let graph = component.graph::<Single>();
            let degrees: Vec<usize> = (0..graph.n()).map(|v| graph.degree(v)).collect();
            Ok::<_, ()>((component.is_grounded(), component.eliminated(), degrees))
        })
        .expect("no component fails");
    assert_eq!(
        built,
        vec![
            (true, 2, vec![2, 1, 1]),
            (false, 1, vec![1, 1]),
            (true, 2, vec![2, 2, 2]),
        ]
    );
}

/// No component holds surplus, so the input stays floating however it was read.
#[test]
fn a_laplacian_carries_no_surplus() {
    let (grounded, _) = sddm(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -1.0, 1.0])
        .map_components(|component| Ok::<_, ()>(component.is_grounded()))
        .expect("no component fails");
    assert_eq!(grounded, [false]);
}
