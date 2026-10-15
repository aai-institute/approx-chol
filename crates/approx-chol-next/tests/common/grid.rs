use crate::graph::{Edge, Graph};

/// Unit-weight `rows × cols` grid Laplacian, row-major.
pub fn grid(rows: usize, cols: usize) -> Graph {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                edges.push(Edge::negative(v, v + 1, 1.0));
            }
            if r + 1 < rows {
                edges.push(Edge::negative(v, v + cols, 1.0));
            }
        }
    }
    Graph::new(rows * cols, edges)
}
