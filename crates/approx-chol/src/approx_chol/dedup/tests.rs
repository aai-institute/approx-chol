use super::*;

fn nbr(to: u32, fill_weight: f64, count: u32) -> Neighbor<f64> {
    Neighbor { to, fill_weight, count }
}

#[test]
fn test_compress_merge_caps_count() {
    let mut dedup = DedupWorkspace::<f64>::new(10);
    let mut raw = vec![
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(3, 1.0, 1),
        nbr(5, 2.0, 1),
    ];
    let mut entries = Vec::new();
    let mut counts = Vec::new();

    dedup.dedup_ac2(&mut raw, &mut entries, &mut counts, 2);

    // Neighbor 3 had 4 copies -> capped to merge_limit=2
    assert_eq!(entries.len(), 2);
    assert_eq!(counts.len(), 2);

    // Find neighbor 3's entry
    let idx3 = entries.iter().position(|e| e.0 == 3).unwrap();
    assert_eq!(counts[idx3], 2);
    // Total weight preserved: 4 copies * 1.0 = 4.0 (count capped, weight unchanged)
    assert!((entries[idx3].1 - 4.0).abs() < 1e-10);

    // Neighbor 5: count=1, weight=2.0
    let idx5 = entries.iter().position(|e| e.0 == 5).unwrap();
    assert_eq!(counts[idx5], 1);
    assert!((entries[idx5].1 - 2.0).abs() < 1e-10);

    // merged_counts should record 2 discarded edges for neighbor 3
    assert_eq!(dedup.merged_counts(), &[(3, 2)]);
}

#[test]
fn test_scatter_ac2_large_multiplicity_caps_without_overflow() {
    let n_edges = 70_000usize;
    let mut dedup = DedupWorkspace::<f64>::new(4);
    let mut raw = vec![nbr(2, 1.0, 1); n_edges];
    let mut entries = Vec::new();
    let mut counts = Vec::new();

    dedup.dedup_ac2(&mut raw, &mut entries, &mut counts, 2);

    assert_eq!(entries, vec![(2, n_edges as f64)]);
    assert_eq!(counts, vec![2]);
    assert_eq!(dedup.merged_counts(), &[(2, (n_edges - 2) as u32)]);
}

#[test]
fn test_virtual_split_plus_fill_edge() {
    let mut dedup = DedupWorkspace::<f64>::new(10);
    // Simulate virtual split edge (count=3, total_weight=6.0) + fill edge (count=1) to same neighbor
    let mut raw = vec![nbr(3, 6.0, 3), nbr(3, 1.5, 1)];
    let mut entries = Vec::new();
    let mut counts = Vec::new();

    dedup.dedup_ac2(&mut raw, &mut entries, &mut counts, 10);

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, 3);
    assert!((entries[0].1 - 7.5).abs() < 1e-10); // 6.0 + 1.5
    assert_eq!(counts[0], 4); // 3 + 1
}

// -----------------------------------------------------------------------
// Equivalence tests: sort path (<= SCATTER_THRESHOLD) vs scatter path (> SCATTER_THRESHOLD)
// -----------------------------------------------------------------------

/// Build a canonical set of raw 3-tuples with `n` unique vertices plus
/// one duplicate pair to exercise merge logic. Vertex IDs stay within `[0, n)`.
fn make_raw_with_duplicate(n: usize) -> Vec<Neighbor<f64>> {
    assert!(n >= 2, "need at least 2 unique vertices");
    let mut raw: Vec<Neighbor<f64>> = (0..n as u32)
        .map(|i| nbr(i, (i + 1) as f64, 1))
        .collect();
    // Add a duplicate for vertex 0 so merged tracking is exercised.
    raw.push(nbr(0, 0.5, 1));
    raw
}

/// AC dedup: sort path (small neighborhood, <= SCATTER_THRESHOLD entries)
/// produces the same deduplicated weights as the scatter path (large neighborhood).
#[test]
fn test_dedup_ac_sort_and_scatter_paths_agree() {
    // Small neighborhood: <= SCATTER_THRESHOLD entries (uses sort path)
    let n_small = SCATTER_THRESHOLD; // exactly at threshold -> sort path
    // Large neighborhood: > SCATTER_THRESHOLD entries (uses scatter path)
    let n_large = SCATTER_THRESHOLD + 10;

    let n_vertices_small = n_small + 1; // for vertex IDs
    let n_vertices_large = n_large + 1;

    let mut dedup_small = DedupWorkspace::<f64>::new(n_vertices_small);
    let mut dedup_large = DedupWorkspace::<f64>::new(n_vertices_large);

    let mut raw_small = make_raw_with_duplicate(n_small);
    let mut raw_large = make_raw_with_duplicate(n_large);

    let mut entries_small: Vec<(u32, f64)> = Vec::new();
    let mut entries_large: Vec<(u32, f64)> = Vec::new();

    dedup_small.dedup_ac(&mut raw_small, &mut entries_small);
    dedup_large.dedup_ac(&mut raw_large, &mut entries_large);

    // Both paths must have merged vertex 0.
    assert!(
        dedup_small.merged().contains(&0),
        "sort path: vertex 0 should be in merged list"
    );
    assert!(
        dedup_large.merged().contains(&0),
        "scatter path: vertex 0 should be in merged list"
    );

    // Both must produce sorted-by-weight output (AC invariant).
    let is_sorted_by_weight = |v: &[(u32, f64)]| {
        v.windows(2)
            .all(|w| w[0].1 < w[1].1 || (w[0].1 == w[1].1 && w[0].0 <= w[1].0))
    };
    assert!(
        is_sorted_by_weight(&entries_small),
        "sort-path AC output must be sorted by weight: {:?}",
        entries_small
    );
    assert!(
        is_sorted_by_weight(&entries_large),
        "scatter-path AC output must be sorted by weight: {:?}",
        entries_large
    );

    // For small path: vertex 0 had weight 1.0 + 0.5 = 1.5 after merge.
    let entry0_small = entries_small.iter().find(|e| e.0 == 0).unwrap();
    assert!(
        (entry0_small.1 - 1.5).abs() < 1e-12,
        "sort path: merged weight for vertex 0 should be 1.5, got {}",
        entry0_small.1
    );

    // For large path: same merge rule applies.
    let entry0_large = entries_large.iter().find(|e| e.0 == 0).unwrap();
    assert!(
        (entry0_large.1 - 1.5).abs() < 1e-12,
        "scatter path: merged weight for vertex 0 should be 1.5, got {}",
        entry0_large.1
    );

    // Each path must have (n - 1) unique entries (n unique raw, one duplicate merged).
    assert_eq!(entries_small.len(), n_small);
    assert_eq!(entries_large.len(), n_large);
}

/// AC2 dedup: sort path and scatter path produce identical results.
///
/// Constructs raw inputs that straddle the SCATTER_THRESHOLD and verifies
/// that both paths:
/// 1. Sum weights correctly for duplicate vertices.
/// 2. Accumulate counts correctly.
/// 3. Apply the merge limit consistently.
/// 4. Sort by average weight (total_weight / count) ascending.
#[test]
fn test_dedup_ac2_sort_and_scatter_paths_agree() {
    // Sort path: n_small raw entries (uses sort-based dedup).
    let n_small = SCATTER_THRESHOLD; // exactly at threshold -> sort path
    // Scatter path: n_large raw entries (uses scatter-gather dedup).
    let n_large = SCATTER_THRESHOLD + 10;

    // Helper: build a canonical raw input with a specific number of entries.
    // Uses n-1 unique vertices (IDs 0..n-2) + one duplicate of vertex 0.
    let make_ac2_raw = |n: usize| -> (Vec<Neighbor<f64>>, usize) {
        let n_unique = n - 1; // one slot used by the duplicate
        let mut raw: Vec<Neighbor<f64>> = (0..n_unique as u32)
            .map(|i| nbr(i, (i + 1) as f64, 2))
            .collect();
        // Duplicate vertex 0 with a different weight and count.
        raw.push(nbr(0, 0.5, 1));
        assert_eq!(raw.len(), n);
        (raw, n_unique)
    };

    let (mut raw_small, n_unique_small) = make_ac2_raw(n_small);
    let (mut raw_large, n_unique_large) = make_ac2_raw(n_large);

    let n_vertices_small = n_unique_small + 1;
    let n_vertices_large = n_unique_large + 1;

    let mut dedup_small = DedupWorkspace::<f64>::new(n_vertices_small);
    let mut dedup_large = DedupWorkspace::<f64>::new(n_vertices_large);

    let merge_limit = 4u32;

    let mut entries_small: Vec<(u32, f64)> = Vec::new();
    let mut counts_small: Vec<u32> = Vec::new();
    let mut entries_large: Vec<(u32, f64)> = Vec::new();
    let mut counts_large: Vec<u32> = Vec::new();

    dedup_small.dedup_ac2(&mut raw_small, &mut entries_small, &mut counts_small, merge_limit);
    dedup_large.dedup_ac2(&mut raw_large, &mut entries_large, &mut counts_large, merge_limit);

    // Both paths: vertex 0 was duplicated, so its merged weight = 1.0 + 0.5 = 1.5
    // and its merged count = 2 + 1 = 3 (within merge_limit=4, no cap).
    let find = |entries: &[(u32, f64)], counts: &[u32], target: u32| {
        entries
            .iter()
            .zip(counts.iter())
            .find(|(&(idx, _), _)| idx == target)
            .map(|(&(_, w), &c)| (w, c))
    };

    let (w0_small, c0_small) = find(&entries_small, &counts_small, 0).unwrap();
    let (w0_large, c0_large) = find(&entries_large, &counts_large, 0).unwrap();

    assert!(
        (w0_small - 1.5).abs() < 1e-12,
        "sort path: merged weight for vertex 0 should be 1.5, got {w0_small}"
    );
    assert_eq!(c0_small, 3, "sort path: merged count for vertex 0 should be 3");

    assert!(
        (w0_large - 1.5).abs() < 1e-12,
        "scatter path: merged weight for vertex 0 should be 1.5, got {w0_large}"
    );
    assert_eq!(c0_large, 3, "scatter path: merged count for vertex 0 should be 3");

    // No merge-limit cap should have fired (count 3 <= limit 4).
    assert!(
        dedup_small.merged_counts().is_empty(),
        "sort path: no entries should be capped at merge_limit=4"
    );
    assert!(
        dedup_large.merged_counts().is_empty(),
        "scatter path: no entries should be capped at merge_limit=4"
    );

    // Entry counts: n_unique deduplicated entries in each case.
    assert_eq!(entries_small.len(), n_unique_small);
    assert_eq!(counts_small.len(), n_unique_small);
    assert_eq!(entries_large.len(), n_unique_large);
    assert_eq!(counts_large.len(), n_unique_large);
}
