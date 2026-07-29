use super::*;
use crate::graph::{Multi, Single, SplitFactor};

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

fn triples<C: EdgeCount>(star: &Star<f64, C>) -> Vec<(u32, f64, u32)> {
    star.entries()
        .iter()
        .map(|entry| (entry.neighbor, entry.weight, entry.copies.get()))
        .collect()
}

/// Folding the merge into [`DegreeDeltas`] would net merge and fill in one clamp,
/// landing vertex 0 at `clamp(2 - 5 + 4) = 1` instead of `clamp(2 - 5) = 0` then
/// `+ 4 = 4`, which flips the pop order below.
#[test]
fn test_merge_floors_immediately_before_batched_fill() {
    let mut ordering = DynamicOrdering::new(&[2, 2], 1);

    // build_star reports 5 merged duplicate edges to vertex 0; applied at once.
    apply_removed_copies(&[(0, 5)], &mut ordering); // 2 - 5 -> floors to 0

    // Same-step fill: +4 to vertex 0, accumulated then flushed as one move.
    let mut deltas = DegreeDeltas::new(2);
    deltas.increase(0, 4); // 0 + 4 -> 4
    deltas.flush(&mut ordering);

    // Vertex 1 (degree 2) outranks vertex 0 (degree 4). A batched merge would
    // make vertex 0 degree 1 and pop it first.
    assert_eq!(ordering.next_vertex(), Some(1));
    assert_eq!(ordering.next_vertex(), Some(0));
}

fn dedup_multi(n: usize, raw: Vec<Neighbor<f64, Multi>>, limit: u32) -> Star<f64, Multi> {
    let mut dedup = DedupWorkspace::<f64, Multi>::new(n);
    dedup.raw = raw;
    let mut star = Star::new();
    dedup.dedup(&mut star, limit);
    star
}

/// Every weight here sums exactly in binary, so the comparison is by value.
#[test]
fn dedup_sums_weights_and_caps_copies() {
    #[allow(clippy::type_complexity)]
    let cases: [(
        &str,
        Vec<Neighbor<f64, Multi>>,
        u32,
        Vec<(u32, f64, u32)>,
        Vec<(u32, u32)>,
    ); 2] = [
        (
            "four single copies capped to two",
            vec![
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(5, 2.0, 1),
            ],
            2,
            // Both average 2.0 per copy, so the tie falls to the lower index.
            vec![(3, 4.0, 2), (5, 2.0, 1)],
            vec![(3, 2)],
        ),
        (
            "virtual split edge plus a fill edge, under the limit",
            vec![nbr(3, 6.0, 3), nbr(3, 1.5, 1)],
            10,
            vec![(3, 7.5, 4)],
            vec![],
        ),
    ];
    for (label, raw, limit, expected, merged) in cases {
        let star = dedup_multi(10, raw, limit);
        assert_eq!(triples(&star), expected, "{label}");
        assert_eq!(star.removed_copies(), merged, "{label}");
    }
}

#[test]
fn test_scatter_large_multiplicity_caps_without_overflow() {
    let n_edges = 70_000usize;
    let raw = vec![nbr(2, 1.0, 1); n_edges];
    let star = dedup_multi(4, raw, 2);

    assert_eq!(triples(&star), vec![(2, n_edges as f64, 2)]);
    assert_eq!(star.removed_copies(), &[(2, (n_edges - 2) as u32)]);
}

/// The limit is not free here: it is `Single`'s own copy count, which is what makes
/// this the AC path.
#[test]
fn a_single_copy_star_keeps_one_copy_and_discards_the_rest() {
    let mut dedup = DedupWorkspace::<f64, Single>::new(3);
    dedup.raw = vec![ac_nbr(2, 3.0), ac_nbr(2, 0.5), ac_nbr(0, 1.0)];
    let mut star = Star::new();
    dedup.dedup(&mut star, Single.get());

    assert_eq!(triples(&star), vec![(0, 1.0, 1), (2, 3.5, 1)]);
    assert_eq!(star.removed_copies(), &[(2, 1)]);
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
fn dedup_single_copy_paths_agree() {
    const LIMIT: u32 = 1;

    let mut by_sort = DedupWorkspace::<f64, Single>::new(3);
    by_sort.raw = ac_raw().to_vec();
    let mut star_sort = Star::new();
    by_sort.dedup_by_sort(&mut star_sort, LIMIT);
    star_sort.sort();

    let mut by_scatter = DedupWorkspace::<f64, Single>::new(3);
    by_scatter.raw = ac_raw().to_vec();
    let mut star_scatter = Star::new();
    by_scatter.dedup_by_scatter(&mut star_scatter, LIMIT);
    star_scatter.sort();

    // Weights summed per vertex, ascending by weight then vertex index.
    assert_eq!(
        triples(&star_sort),
        vec![(0, 1.25, 1), (2, 3.5, 1), (1, 4.0, 1)]
    );
    assert_eq!(triples(&star_scatter), triples(&star_sort));

    assert_eq!(
        sorted_merged(star_sort.removed_copies()),
        vec![(0, 1), (2, 1)]
    );
    assert_eq!(
        sorted_merged(star_scatter.removed_copies()),
        sorted_merged(star_sort.removed_copies())
    );
}

/// Every other star test uses distinct keys, so a reversed tie-break would change the
/// clique-tree path with nothing to notice.
#[test]
fn equal_sort_keys_order_by_ascending_neighbor() {
    let mut single = Star::<f64, Single>::new();
    for neighbor in [5, 2, 9] {
        single.push(StarEntry {
            neighbor,
            copies: Single,
            weight: 1.5,
        });
    }
    single.sort();
    assert_eq!(
        triples(&single),
        vec![(2, 1.5, 1), (5, 1.5, 1), (9, 1.5, 1)]
    );

    // The multi-copy branch sorts through `sort_scratch` on the per-copy quotient,
    // so it needs its own tie: every quotient here is 1.5 and exact in binary.
    let mut multi = Star::<f64, Multi>::new();
    for (neighbor, weight, copies) in [(5u32, 3.0, 2u32), (2, 1.5, 1), (9, 6.0, 4)] {
        multi.push(StarEntry {
            neighbor,
            copies: Multi::new(copies),
            weight,
        });
    }
    multi.sort();
    assert_eq!(triples(&multi), vec![(2, 1.5, 1), (5, 3.0, 2), (9, 6.0, 4)]);
}

#[test]
fn dedup_multi_copy_paths_agree() {
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

    let mut by_sort = DedupWorkspace::<f64, Multi>::new(3);
    by_sort.raw = raw().to_vec();
    let mut star_sort = Star::new();
    by_sort.dedup_by_sort(&mut star_sort, LIMIT);
    star_sort.sort();

    let mut by_scatter = DedupWorkspace::<f64, Multi>::new(3);
    by_scatter.raw = raw().to_vec();
    let mut star_scatter = Star::new();
    by_scatter.dedup_by_scatter(&mut star_scatter, LIMIT);
    star_scatter.sort();

    // Weights and copies summed per vertex, vertex 2 capped from 5 to LIMIT,
    // ascending by weight/copies: 1.25/2 < 3.5/4 < 4.0/2.
    assert_eq!(
        triples(&star_sort),
        vec![(0, 1.25, 2), (2, 3.5, 4), (1, 4.0, 2)]
    );
    assert_eq!(triples(&star_scatter), triples(&star_sort));

    assert_eq!(sorted_merged(star_sort.removed_copies()), vec![(2, 1)]);
    assert_eq!(
        sorted_merged(star_scatter.removed_copies()),
        sorted_merged(star_sort.removed_copies())
    );
}

/// AC's entry must not pay for a count its layout knows statically.
#[test]
fn a_single_copy_star_entry_is_as_wide_as_the_bare_pair() {
    assert_eq!(
        size_of::<StarEntry<f64, Single>>(),
        size_of::<(u32, f64)>(),
        "f64 entry"
    );
    assert_eq!(
        size_of::<StarEntry<f32, Single>>(),
        size_of::<(u32, f32)>(),
        "f32 entry"
    );
    assert_eq!(
        size_of::<StarEntry<f64, Multi>>(),
        size_of::<StarEntry<f64, Single>>(),
        "the multi count lands in the pair's padding"
    );
}

#[test]
fn the_split_is_the_cap_and_the_surviving_copy_count() {
    let k = SplitFactor::new(3).expect("3 splits");
    let mut dedup = DedupWorkspace::<f64, Multi>::new(4);
    dedup.raw = vec![nbr(1, 3.0, 3), nbr(1, 3.0, 3)];
    let mut star = Star::new();
    dedup.dedup(&mut star, k.get());

    assert_eq!(triples(&star), vec![(1, 6.0, 3)]);
    assert_eq!(star.removed_copies(), &[(1, 3)]);
}
