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

/// Disjoint off-diagonal graphs: read from the CSR alone, connectivity splits them.
#[test]
fn components_sharing_a_ground_are_one_block() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
    );
    assert!(
        blocks.is_none(),
        "a shared ground joins every grounded component, got {blocks:?}"
    );
}

/// No surplus grounds them, so a layout that merged unconditionally fails here.
#[test]
fn components_with_no_surplus_stay_separate() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 1], vec![2, 3]]));
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

/// The layout precedes any graph, so this pins that a block's graph has the vertices it promised.
#[test]
fn only_a_grounded_component_appends_a_ground_slot() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let col_indices = [0u32, 1, 0, 1, 2, 3, 2, 3];
    let values = [5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4).expect("valid CSR");
    let sddm = Sddm::try_from(csr).expect("valid SDDM");

    let layout = components(&sddm).expect("two blocks");
    let blocks: Vec<Vec<u32>> = layout.blocks().map(<[u32]>::to_vec).collect();
    assert_eq!(blocks, vec![vec![0, 1], vec![2, 3]]);

    let mut local_of = vec![0u32; sddm.n()];
    let built: Vec<(usize, usize, bool)> = blocks
        .iter()
        .map(|vertices| {
            let component = Component::new(&sddm, BlockVertices::part(vertices, &mut local_of));
            (
                component.graph::<Single>().n(),
                component.eliminated(),
                component.is_grounded(),
            )
        })
        .collect();
    assert_eq!(built, vec![(3, 2, true), (2, 1, false)]);
}
