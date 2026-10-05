use super::*;
use crate::graph::Single;
use crate::{CsrRef, Sddm};

fn ingestion_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Ingestion<f64> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    Ingestion::of(Sddm::try_from(csr).expect("valid SDDM"))
}

/// Blocks are what the layout says they are, in its own numbering.
fn blocks_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Option<Vec<Vec<u32>>> {
    ingestion_of(row_ptrs, col_indices, values)
        .take_layout()
        .map(|layout| layout.blocks().map(<[u32]>::to_vec).collect::<Vec<_>>())
}

/// Each component grounds itself, so surplus never joins two of them.
#[test]
fn grounded_components_stay_separate() {
    let blocks = blocks_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
    );
    assert_eq!(blocks, Some(vec![vec![0, 1], vec![2, 3]]));
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

/// One grounded component beside a floating one: only the grounded one's graph gains
/// a ground vertex.
#[test]
fn only_a_grounded_block_gets_a_ground_vertex() {
    let mut ingestion = ingestion_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0],
    );
    let layout = ingestion.take_layout().expect("two blocks");
    let blocks: Vec<Vec<u32>> = layout.blocks().map(<[u32]>::to_vec).collect();
    assert_eq!(blocks, vec![vec![0, 1], vec![2, 3]]);

    let mut local_of = vec![0u32; ingestion.n()];
    let built: Vec<(bool, usize)> = blocks
        .iter()
        .map(|vertices| {
            let view = BlockVertices::part(vertices, &mut local_of);
            let grounded = ingestion.is_grounded(&view);
            (
                grounded,
                ingestion.block_graph::<Single>(&view, grounded).n(),
            )
        })
        .collect();
    assert_eq!(built, vec![(true, 3), (false, 2)]);
}
