use std::collections::BTreeSet;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::csr::OwnedCsr;

/// Barabási–Albert graph Laplacian: each new vertex attaches to `m` targets drawn by degree.
pub fn barabasi_albert(n: usize, m: usize, seed: u64) -> OwnedCsr {
    assert!(m >= 1 && n > m);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut adj: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n];
    let mut degree_list: Vec<u32> = Vec::new();

    for i in 0..m {
        for j in (i + 1)..m {
            adj[i].insert(j as u32);
            adj[j].insert(i as u32);
            degree_list.push(i as u32);
            degree_list.push(j as u32);
        }
    }

    for v in m..n {
        let mut targets = BTreeSet::new();
        while targets.len() < m {
            if degree_list.is_empty() {
                let t = rng.random_range(0..v as u32);
                targets.insert(t);
            } else {
                let idx = rng.random_range(0..degree_list.len());
                targets.insert(degree_list[idx]);
            }
        }
        for &t in &targets {
            adj[v].insert(t);
            adj[t as usize].insert(v as u32);
            degree_list.push(v as u32);
            degree_list.push(t);
        }
    }

    OwnedCsr::laplacian(
        adj.iter()
            .map(|neighbors| neighbors.iter().map(|&u| u as usize)),
    )
}
