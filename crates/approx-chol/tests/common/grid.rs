use crate::graph::{Edge, Graph};

/// Unit-weight `rows × cols` grid Laplacian, row-major: every horizontal edge, then every vertical one.
pub fn grid(rows: usize, cols: usize) -> Graph {
    let right = (0..rows * cols)
        .filter(|v| v % cols + 1 < cols)
        .map(|v| (v, v + 1));
    let down = (0..cols * rows.saturating_sub(1)).map(|v| (v, v + cols));
    let edges = right.chain(down).map(|(a, b)| Edge::negative(a, b, 1.0));
    Graph::new(rows * cols, edges.collect())
}
