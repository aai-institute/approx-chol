use super::graph::{Edge, Graph};

/// Unit-weight `rows × cols` grid Laplacian, row-major: every horizontal edge, then every vertical one.
pub fn grid(rows: usize, cols: usize) -> Graph {
    let right = (0..rows * cols)
        .filter(|v| v % cols + 1 < cols)
        .map(|v| (v, v + 1));
    let down = (0..cols * rows.saturating_sub(1)).map(|v| (v, v + cols));
    let edges = right.chain(down).map(|(a, b)| Edge::negative(a, b, 1.0));
    Graph::new(rows * cols, edges.collect())
}

/// A frustrated `k × k` grid, every seventh edge positive, row 0's edges `light` and their ends grounded by `0.01·light` each: its surplus is `~1e-3·light` of the trace.
pub fn light_row(k: usize, light: f64) -> Graph {
    let edges = (grid(k, k).edges.into_iter().enumerate()).map(|(e, edge)| Edge {
        weight: if edge.a < k { light } else { 1.0 },
        positive: e % 7 == 0,
        ..edge
    });
    let mut graph = Graph::new(k * k, edges.collect());
    for edge in &graph.edges {
        for v in [edge.a, edge.b].into_iter().filter(|&v| v < k) {
            graph.ground[v] += 0.01 * edge.weight;
        }
    }
    graph
}
