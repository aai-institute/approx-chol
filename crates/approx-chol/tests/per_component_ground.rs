use approx_chol::{factorize_with, Backend, Config, CsrRef};
use rstest::rstest;

/// Laplacian edges plus diagonal surplus, stored as a full symmetric CSR.
struct Sddm {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<f64>,
}

impl Sddm {
    fn new(n: usize, edges: &[(usize, usize, f64)], surplus: &[(usize, f64)]) -> Self {
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut diagonal = vec![0.0; n];
        for &(a, b, weight) in edges {
            rows[a].push((b as u32, -weight));
            rows[b].push((a as u32, -weight));
            diagonal[a] += weight;
            diagonal[b] += weight;
        }
        for &(vertex, s) in surplus {
            diagonal[vertex] += s;
        }
        let mut sddm = Self {
            row_ptrs: vec![0],
            col_indices: Vec::new(),
            values: Vec::new(),
        };
        for (row, mut entries) in rows.into_iter().enumerate() {
            entries.push((row as u32, diagonal[row]));
            entries.sort_by_key(|&(col, _)| col);
            for (col, value) in entries {
                sddm.col_indices.push(col);
                sddm.values.push(value);
            }
            sddm.row_ptrs.push(sddm.col_indices.len() as u32);
        }
        sddm
    }

    fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    fn solve(&self, b: &[f64], config: Config) -> Vec<f64> {
        let csr = CsrRef::new(
            &self.row_ptrs,
            &self.col_indices,
            &self.values,
            self.n() as u32,
        )
        .expect("valid CSR");
        factorize_with(csr, config)
            .expect("factorization")
            .solve(b)
            .expect("solve")
    }

    fn residual(&self, x: &[f64], b: &[f64]) -> f64 {
        let worst = (0..self.n())
            .map(|row| {
                let range = self.row_ptrs[row] as usize..self.row_ptrs[row + 1] as usize;
                let product: f64 = range
                    .map(|k| self.values[k] * x[self.col_indices[k] as usize])
                    .sum();
                (product - b[row]).abs()
            })
            .fold(0.0, f64::max);
        worst / b.iter().fold(0.0, |max: f64, v| max.max(v.abs()))
    }
}

fn complete(vertices: &[usize], scale: f64) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for (i, &a) in vertices.iter().enumerate() {
        for (j, &b) in vertices.iter().enumerate().skip(i + 1) {
            edges.push((a, b, scale * (1.0 + 0.25 * ((i + 2 * j) % 5) as f64)));
        }
    }
    edges
}

fn path(vertices: &[usize]) -> Vec<(usize, usize, f64)> {
    vertices
        .windows(2)
        .enumerate()
        .map(|(i, pair)| (pair[0], pair[1], 1.0 + 0.5 * (i % 3) as f64))
        .collect()
}

/// `count` components of `size` vertices, the `k`-th on vertices `k, k + count, ...`.
fn interleaved(count: usize, size: usize) -> Vec<Vec<usize>> {
    (0..count)
        .map(|k| (0..size).map(|i| i * count + k).collect())
        .collect()
}

/// A shared ground let sampled fill carry one component's right-hand side into the others.
#[rstest]
#[case::approximate(Backend::Approximate, None)]
#[case::approximate_split(Backend::Approximate, Some(2))]
#[case::default(Backend::default(), None)]
fn a_right_hand_side_on_one_component_leaves_the_others_at_zero(
    #[case] backend: Backend,
    #[case] split_merge: Option<u32>,
) {
    let parts = interleaved(3, 5);
    let edges: Vec<_> = parts.iter().flat_map(|part| complete(part, 1.0)).collect();
    // The third component floats.
    let sddm = Sddm::new(15, &edges, &[(parts[0][0], 0.5), (parts[1][2], 0.5)]);
    let mut b = vec![0.0; 15];
    for (i, &vertex) in parts[0].iter().enumerate() {
        b[vertex] = 1.0 + i as f64;
    }
    let config = Config {
        backend,
        split_merge,
        ..Config::default()
    };

    let x = sddm.solve(&b, config);
    for &vertex in parts[1].iter().chain(&parts[2]) {
        assert_eq!(x[vertex], 0.0, "vertex {vertex}");
    }
}

/// Each fits the dense backend alone; together with a shared ground they did not.
#[test]
fn small_grounded_components_route_exact_one_by_one() {
    let parts = interleaved(2, 13);
    let edges: Vec<_> = parts.iter().flat_map(|part| complete(part, 1.0)).collect();
    let sddm = Sddm::new(26, &edges, &[(parts[0][0], 0.5), (parts[1][7], 2.0)]);
    let b: Vec<f64> = (0..26).map(|i| (i % 7) as f64 - 3.0).collect();

    let x = sddm.solve(&b, Config::default());
    assert!(sddm.residual(&x, &b) < 1e-13, "{}", sddm.residual(&x, &b));
}

/// A path grounded at one end stays a tree, which elimination factors exactly.
#[rstest]
#[case::approximate(Backend::Approximate)]
#[case::default(Backend::default())]
fn interleaved_grounded_paths_each_solve_exactly(#[case] backend: Backend) {
    let parts = interleaved(2, 32);
    let edges: Vec<_> = parts.iter().flat_map(|part| path(part)).collect();
    let sddm = Sddm::new(64, &edges, &[(parts[0][0], 1.0), (parts[1][31], 0.25)]);
    let b: Vec<f64> = (0..64).map(|i| (i % 5) as f64 - 1.5).collect();
    let config = Config {
        backend,
        ..Config::default()
    };

    let x = sddm.solve(&b, config);
    assert!(sddm.residual(&x, &b) < 1e-12, "{}", sddm.residual(&x, &b));
}

/// A power of two scales every factor entry exactly, so a component's solve rescales bit for bit.
#[test]
fn scaling_one_grounded_component_rescales_only_its_solution() {
    let scale = f64::powi(2.0, -40);
    let parts = interleaved(2, 5);
    let build = |second: f64| {
        let mut edges = complete(&parts[0], 1.0);
        edges.extend(complete(&parts[1], second));
        Sddm::new(
            10,
            &edges,
            &[(parts[0][1], 0.5), (parts[1][3], 0.5 * second)],
        )
    };
    let b: Vec<f64> = (0..10).map(|i| (i % 4) as f64 - 1.0).collect();
    let config = Config {
        backend: Backend::Approximate,
        ..Config::default()
    };

    let unit = build(1.0).solve(&b, config);
    let scaled = build(scale).solve(&b, config);
    for &vertex in &parts[0] {
        assert_eq!(scaled[vertex], unit[vertex], "vertex {vertex}");
    }
    for &vertex in &parts[1] {
        assert_eq!(scaled[vertex] * scale, unit[vertex], "vertex {vertex}");
    }
}

fn grid(side: usize, offset: usize) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    for r in 0..side {
        for c in 0..side {
            let v = offset + r * side + c;
            if c + 1 < side {
                edges.push((v, v + 1, 1.0));
            }
            if r + 1 < side {
                edges.push((v, v + side, 1.0));
            }
        }
    }
    edges
}

/// Only separate grounds let the dense backend claim one grounded component and sample the other.
#[test]
fn claiming_a_small_grounded_component_leaves_a_large_ones_draws() {
    let mut edges = grid(3, 0);
    edges.extend(grid(7, 9));
    let sddm = Sddm::new(58, &edges, &[(4, 0.5), (9 + 24, 0.5)]);
    let b: Vec<f64> = (0..58).map(|i| (i % 6) as f64 - 2.5).collect();
    let solve = |backend| {
        sddm.solve(
            &b,
            Config {
                backend,
                ..Config::default()
            },
        )
    };

    let approximated = solve(Backend::Approximate);
    let claimed = solve(Backend::default());
    assert_eq!(approximated[9..], claimed[9..]);
    assert_ne!(
        approximated[..9],
        claimed[..9],
        "the small component went exact"
    );
}
