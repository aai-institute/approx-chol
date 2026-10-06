use super::*;
use crate::graph::{Multi, Single, SplitFactor};

fn split(k: u32) -> SplitFactor {
    SplitFactor::new(k).expect("the fixtures split by 2 or more")
}

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

/// A batched merge would land vertex 0 at `clamp(2 - 5 + 4) = 1`, not `0 + 4`, flipping the pop.
#[test]
fn test_merge_floors_immediately_before_batched_fill() {
    let mut ordering = DynamicOrdering::new(&[2, 2], 1);

    apply_removed_copies(&[(0, 5)], &mut ordering);

    let mut deltas = DegreeDeltas::new(2);
    deltas.increase(0, 4);
    deltas.flush(&mut ordering);

    assert_eq!(ordering.next_vertex(), Some(1));
    assert_eq!(ordering.next_vertex(), Some(0));
}

fn dedup_multi(n: usize, raw: Vec<Neighbor<f64, Multi>>, limit: SplitFactor) -> Star<f64, Multi> {
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
        SplitFactor,
        Vec<(u32, f64, u32)>,
        Vec<(u32, u32)>,
    ); 3] = [
        (
            "four single copies capped to two",
            vec![
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(3, 1.0, 1),
                nbr(5, 2.0, 1),
            ],
            split(2),
            // Both average 2.0 per copy, so the tie falls to the lower index.
            vec![(3, 4.0, 2), (5, 2.0, 1)],
            vec![(3, 2)],
        ),
        (
            "virtual split edge plus a fill edge, under the limit",
            vec![nbr(3, 6.0, 3), nbr(3, 1.5, 1)],
            split(10),
            vec![(3, 7.5, 4)],
            vec![],
        ),
        (
            "two split edges merged past the split",
            vec![nbr(1, 3.0, 3), nbr(1, 3.0, 3)],
            split(3),
            vec![(1, 6.0, 3)],
            vec![(1, 3)],
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
    let star = dedup_multi(4, raw, split(2));

    assert_eq!(triples(&star), vec![(2, n_edges as f64, 2)]);
    assert_eq!(star.removed_copies(), &[(2, (n_edges - 2) as u32)]);
}

// `dedup` routes by length, so these call each path directly; merge orders differ by path.

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
    let mut by_sort = DedupWorkspace::<f64, Single>::new(3);
    by_sort.raw = ac_raw().to_vec();
    let mut star_sort = Star::new();
    by_sort.dedup_by_sort(&mut star_sort, ());
    star_sort.sort();

    let mut by_scatter = DedupWorkspace::<f64, Single>::new(3);
    by_scatter.raw = ac_raw().to_vec();
    let mut star_scatter = Star::new();
    by_scatter.dedup_by_scatter(&mut star_scatter, ());
    star_scatter.sort();

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

/// Other star tests use distinct keys, so a reversed tie-break would go unnoticed.
#[test]
fn equal_sort_keys_order_by_ascending_neighbor() {
    let mut single = Star::<f64, Single>::new();
    for neighbor in [5, 2, 9] {
        single.entries.push(StarEntry {
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

    // The multi-copy branch sorts by per-copy quotient, so it needs its own tie (all 1.5).
    let mut multi = Star::<f64, Multi>::new();
    for (neighbor, weight, copies) in [(5u32, 3.0, 2u32), (2, 1.5, 1), (9, 6.0, 4)] {
        multi.entries.push(StarEntry {
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
    let limit = split(4);
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
    by_sort.dedup_by_sort(&mut star_sort, limit);
    star_sort.sort();

    let mut by_scatter = DedupWorkspace::<f64, Multi>::new(3);
    by_scatter.raw = raw().to_vec();
    let mut star_scatter = Star::new();
    by_scatter.dedup_by_scatter(&mut star_scatter, limit);
    star_scatter.sort();

    // Vertex 2 caps from 5 to 4; ordered by weight/copies: 1.25/2 < 3.5/4 < 4.0/2.
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
