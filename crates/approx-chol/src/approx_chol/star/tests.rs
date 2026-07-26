use super::*;
use crate::test_utils::OrPanic;

fn nbr(to: u32, fill_weight: f64, count: u32) -> Neighbor<f64, Multi> {
    Neighbor {
        to,
        fill_weight,
        count: Multi::new(count),
    }
}

fn ac_nbr(to: u32, fill_weight: f64) -> Neighbor<f64, Single> {
    Neighbor {
        to,
        fill_weight,
        count: Single,
    }
}

/// Pins the per-step degree-update protocol: merge-compression decrements are
/// applied to the ordering *immediately* (and floor at zero) by
/// `apply_merged_counts`, **before** the step's fill/removal net delta is
/// batched through [`DegreeDeltas`] and flushed. The sub-zero excess of a merge
/// must therefore be lost, not offset later fill in the same step.
///
/// If a refactor folded the merge into `DegreeDeltas`, the merge and fill would
/// net in one clamp and vertex 0's estimate would land at `clamp(2 - 5 + 4) = 1`
/// instead of `clamp(2 - 5) = 0` then `0 + 4 = 4` — flipping the pop order below.
#[test]
fn test_merge_floors_immediately_before_batched_fill() {
    let mut ordering = DynamicOrdering::new(&[2, 2], 1);

    // build_star reports 5 merged duplicate edges to vertex 0; applied at once.
    apply_merged_counts(&[(0, 5)], &mut ordering); // 2 - 5 -> floors to 0

    // Same-step fill: +4 to vertex 0, accumulated then flushed as one move.
    let mut deltas = DegreeDeltas::new(2);
    deltas.increase(0, 4); // 0 + 4 -> 4
    deltas.flush(&mut ordering);

    // Vertex 1 (degree 2) outranks vertex 0 (degree 4). A batched merge would
    // make vertex 0 degree 1 and pop it first.
    assert_eq!(ordering.next_vertex(), Some(1));
    assert_eq!(ordering.next_vertex(), Some(0));
}

/// `(weight, count)` for `neighbor`, or `None` if the star has no such entry.
fn find(star: &MultiStar<f64>, neighbor: u32) -> Option<(f64, u32)> {
    star.iter()
        .find(|&(n, _, _)| n == neighbor)
        .map(|(_, weight, count)| (weight, count))
}

fn dedup_ac2(
    n: usize,
    raw: &mut [Neighbor<f64, Multi>],
    merge_limit: u32,
) -> (Ac2DedupWorkspace<f64>, MultiStar<f64>) {
    let mut dedup = Ac2DedupWorkspace::<f64>::new(n);
    let mut star = MultiStar::new();
    dedup.dedup(raw, &mut star, merge_limit);
    (dedup, star)
}

#[test]
fn test_compress_merge_caps_count() {
    let mut raw = vec![
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(5, 2.0, 1),
    ];
    let (dedup, star) = dedup_ac2(10, &mut raw, 2);

    // Neighbor 3 had 4 copies -> count capped to merge_limit=2, weight preserved.
    let (w3, c3) = find(&star, 3).or_panic("missing entry for neighbor 3");
    assert_eq!(c3, 2);
    assert!((w3 - 4.0).abs() < 1e-10);

    let (w5, c5) = find(&star, 5).or_panic("missing entry for neighbor 5");
    assert_eq!(c5, 1);
    assert!((w5 - 2.0).abs() < 1e-10);

    assert_eq!(star.entries().len(), 2);
    // merged_counts should record 2 discarded edges for neighbor 3
    assert_eq!(dedup.merged_counts(), &[(3, 2)]);
}

#[test]
fn test_scatter_ac2_large_multiplicity_caps_without_overflow() {
    let n_edges = 70_000usize;
    let mut raw = vec![nbr(2, 1.0, 1); n_edges];
    let (dedup, star) = dedup_ac2(4, &mut raw, 2);

    assert_eq!(
        star.iter().collect::<Vec<_>>(),
        vec![(2, n_edges as f64, 2)]
    );
    assert_eq!(dedup.merged_counts(), &[(2, (n_edges - 2) as u32)]);
}

#[test]
fn test_virtual_split_plus_fill_edge() {
    // Virtual split edge (count=3, total_weight=6.0) + fill edge (count=1) to
    // the same neighbor.
    let mut raw = vec![nbr(3, 6.0, 3), nbr(3, 1.5, 1)];
    let (_, star) = dedup_ac2(10, &mut raw, 10);

    assert_eq!(star.iter().collect::<Vec<_>>(), vec![(3, 7.5, 4)]);
}

// -----------------------------------------------------------------------
// Equivalence: sort path (<= SCATTER_THRESHOLD) vs scatter path (above it)
//
// `dedup` dispatches on `raw.len()`, so no single input can reach both paths
// through it; these call each path directly instead. The paths report merges in
// different orders (neighbor-sorted vs first-seen), so only the sorted merge
// lists are compared. Both fixtures use weights whose duplicate sums are exact
// in binary — the paths accumulate in different orders, which is not the claim.
// -----------------------------------------------------------------------

fn sorted_merged(merged: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out = merged.to_vec();
    out.sort_unstable();
    out
}

/// Raw neighborhood with two duplicated vertices and one singleton.
fn ac_raw() -> [Neighbor<f64, Single>; 5] {
    [
        ac_nbr(2, 3.0),
        ac_nbr(0, 1.0),
        ac_nbr(2, 0.5),
        ac_nbr(1, 4.0),
        ac_nbr(0, 0.25),
    ]
}

#[test]
fn dedup_ac_paths_agree() {
    let mut by_sort = AcDedupWorkspace::<f64>::new(3);
    let mut sorted_entries = Vec::new();
    by_sort.dedup_sort_small(&mut ac_raw(), &mut sorted_entries);

    let mut by_scatter = AcDedupWorkspace::<f64>::new(3);
    let mut scattered_entries = Vec::new();
    by_scatter.dedup_scatter(&ac_raw(), &mut scattered_entries);

    // Weights summed per vertex, ascending by weight then vertex index.
    assert_eq!(sorted_entries, vec![(0, 1.25), (2, 3.5), (1, 4.0)]);
    assert_eq!(scattered_entries, sorted_entries);

    assert_eq!(sorted_merged(by_sort.merged_counts()), vec![(0, 1), (2, 1)]);
    assert_eq!(
        sorted_merged(by_scatter.merged_counts()),
        sorted_merged(by_sort.merged_counts())
    );
}

#[test]
fn dedup_ac2_paths_agree() {
    const LIMIT: u32 = 4;
    let raw = || {
        [
            nbr(2, 3.0, 2),
            nbr(0, 1.0, 1),
            nbr(2, 0.5, 3),
            nbr(1, 4.0, 2),
            nbr(0, 0.25, 1),
        ]
    };

    let mut by_sort = Ac2DedupWorkspace::<f64>::new(3);
    let mut star_sort = MultiStar::new();
    by_sort.dedup_sort(&mut raw(), &mut star_sort);
    star_sort.apply_merge_limit(LIMIT, &mut by_sort.merged_counts);
    star_sort.sort_by_avg_weight();

    let mut by_scatter = Ac2DedupWorkspace::<f64>::new(3);
    let mut star_scatter = MultiStar::new();
    by_scatter.dedup_scatter(&raw(), &mut star_scatter);
    star_scatter.apply_merge_limit(LIMIT, &mut by_scatter.merged_counts);
    star_scatter.sort_by_avg_weight();

    // Weights and counts summed per vertex, vertex 2 capped from 5 to LIMIT,
    // ascending by weight/count: 1.25/2 < 3.5/4 < 4.0/2.
    assert_eq!(
        star_sort.iter().collect::<Vec<_>>(),
        vec![(0, 1.25, 2), (2, 3.5, 4), (1, 4.0, 2)]
    );
    assert_eq!(
        star_scatter.iter().collect::<Vec<_>>(),
        star_sort.iter().collect::<Vec<_>>()
    );

    assert_eq!(sorted_merged(&by_sort.merged_counts), vec![(2, 1)]);
    assert_eq!(
        sorted_merged(&by_scatter.merged_counts),
        sorted_merged(&by_sort.merged_counts)
    );
}
