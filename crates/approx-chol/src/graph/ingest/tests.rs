use super::*;
use crate::graph::Single;

/// Blocks are what the layout says they are, in its own numbering.
fn blocks_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Option<Vec<Vec<u32>>> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    let canonical = Canonical::of(csr).expect("canonical");
    validate(&canonical)
        .expect("valid SDDM")
        .layout
        .map(|layout| layout.blocks().map(<[u32]>::to_vec).collect::<Vec<_>>())
}

/// Disjoint off-diagonal graphs: read from the CSR alone, connectivity splits them.
#[test]
fn components_sharing_a_ground_vertex_are_one_block() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
    );
    assert!(
        blocks.is_none(),
        "a shared ground vertex makes the augmented graph connected, got {blocks:?}"
    );
}

/// No surplus, so nothing grounds them — the test above would pass on a layout
/// that merged unconditionally.
#[test]
fn components_with_no_surplus_stay_separate() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 1], vec![2, 3]]));
}

/// One grounded component beside a floating one. The ground vertex outranks every
/// real vertex, so it lands last in the block it joins rather than opening one.
#[test]
fn the_ground_vertex_lands_last_in_its_own_block() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 1, 4], vec![2, 3]]));
}

/// A vertex no edge reaches is its own block, which is what makes the ordering
/// "by lowest member" observable rather than incidental.
#[test]
fn blocks_are_ordered_by_their_lowest_vertex() {
    let blocks = blocks_of(
        &[0, 2, 3, 4, 6],
        &[0, 3, 1, 2, 0, 3],
        &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 3], vec![1], vec![2]]));
}

/// The layout is read before any graph exists, so this pins that the graph a block
/// is actually handed still has the vertices the layout promised it.
#[test]
fn the_built_graph_agrees_with_the_layout() {
    let row_ptrs = [0u32, 2, 4, 6, 8];
    let col_indices = [0u32, 1, 0, 1, 2, 3, 2, 3];
    let values = [5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0];
    let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4).expect("valid CSR");
    let mut ingestion = Ingestion::of(csr).expect("valid SDDM");
    assert_eq!(ingestion.n(), 5);

    let layout = ingestion.take_layout().expect("two blocks");
    let blocks: Vec<Vec<u32>> = layout.blocks().map(<[u32]>::to_vec).collect();
    assert_eq!(blocks, vec![vec![0, 1, 4], vec![2, 3]]);

    let mut local_of = vec![0u32; ingestion.n()];
    let built: Vec<(usize, bool)> = blocks
        .iter()
        .map(|vertices| {
            let view = BlockVertices::part(vertices, &mut local_of);
            let carries = ingestion.carries_ground(&view);
            (ingestion.block_graph::<Single>(&view).n(), carries)
        })
        .collect();
    assert_eq!(
        built,
        vec![(3, true), (2, false)],
        "only the block ending at the ground vertex is anchored to it"
    );
}
