use super::*;
use crate::graph::Single;
use crate::CsrRef;

fn sddm_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Sddm<f64> {
    let n = (row_ptrs.len() - 1) as u32;
    let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
    Sddm::try_from(csr).expect("valid SDDM")
}

/// Each component's input vertices, in layout order.
fn vertices_of(sddm: &Sddm<f64>) -> Vec<Vec<usize>> {
    Components::of(sddm)
        .iter()
        .map(|component| {
            let view = component.view();
            (0..view.vertices.len())
                .map(|local| view.global(local))
                .collect()
        })
        .collect()
}

/// Each component grounds itself, so surplus never joins two of them.
#[test]
fn grounded_components_stay_separate() {
    let sddm = sddm_of(
        &[0, 2, 4, 6, 8],
        &[0, 1, 0, 1, 2, 3, 2, 3],
        &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
    );
    assert_eq!(vertices_of(&sddm), [[0, 1], [2, 3]]);
}

/// A vertex no edge reaches is its own component, which is what makes the ordering
/// "by lowest member" observable rather than incidental.
#[test]
fn components_are_ordered_by_their_lowest_vertex() {
    let sddm = sddm_of(
        &[0, 2, 3, 4, 6],
        &[0, 3, 1, 2, 0, 3],
        &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0],
    );
    assert_eq!(vertices_of(&sddm), [vec![0, 3], vec![1], vec![2]]);
}

/// Interleaved grounded and floating components: each reads its rows in local numbering.
#[test]
fn each_component_is_classified_and_renumbered() {
    let sddm = sddm_of(
        &[0, 2, 4, 6, 8],
        &[0, 2, 1, 3, 0, 2, 1, 3],
        &[5.0, -1.0, 1.0, -1.0, -1.0, 4.0, -1.0, 1.0],
    );
    let components = Components::of(&sddm);
    let [grounded, floating] = <[_; 2]>::try_from(components.iter().collect::<Vec<_>>())
        .ok()
        .expect("two components");

    assert!(matches!(grounded, Component::Grounded { .. }));
    assert_eq!(grounded.eliminated(), 2);
    assert_eq!(grounded.graph::<Single>().n(), 3);
    let mut row = Vec::new();
    grounded
        .view()
        .upper_row(0, |col, weight| row.push((col, weight)));
    assert_eq!(row, [(1, 1.0)]);

    assert!(matches!(floating, Component::Laplacian(_)));
    assert_eq!(floating.eliminated(), 1);
    assert_eq!(floating.graph::<Single>().n(), 2);

    assert_eq!(components.into_order(), Some(vec![0, 2, 1, 3]));
}
