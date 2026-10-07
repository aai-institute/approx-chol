use super::*;
use crate::graph::Single;
use crate::CsrRef;

/// Blocks are what the layout says they are, in its own numbering.
fn blocks_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Option<Vec<Vec<u32>>> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    let sddm = Sddm::try_from(csr).expect("valid SDDM");
    components(&sddm).map(|layout| layout.blocks().map(<[u32]>::to_vec).collect::<Vec<_>>())
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

/// Each grounded component gets its own ground slot, joined to its own surplus vertices only.
#[test]
fn each_grounded_component_appends_its_own_ground_slot() {
    let csr = CsrRef::new(&ROW_PTRS, &COL_INDICES, &VALUES, 6).expect("valid CSR");
    let sddm = Sddm::try_from(csr).expect("valid SDDM");
    let layout = components(&sddm).expect("three blocks");

    let mut local_of = vec![0u32; sddm.n()];
    let built: Vec<(bool, usize, Vec<usize>)> = layout
        .blocks()
        .map(|vertices| {
            let component = Component::new(&sddm, BlockVertices::part(vertices, &mut local_of));
            let graph = component.graph::<Single>();
            let degrees = (0..graph.n()).map(|v| graph.degree(v)).collect();
            (component.is_grounded(), component.eliminated(), degrees)
        })
        .collect();
    assert_eq!(
        built,
        vec![
            (true, 2, vec![2, 1, 1]),
            (false, 1, vec![1, 1]),
            (true, 2, vec![2, 2, 2]),
        ]
    );
}
