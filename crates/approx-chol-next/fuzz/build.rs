use std::fs;
use std::path::Path;

use approx_chol_next::BlockFactor;

mod case;

use case::{Build, Case, Input, Rows, Side, View, Weighting};

/// Written at build time rather than committed, so a seed never outlives the encoding of `Input` or `BlockFactor`.
fn main() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let (views, factors) = (
        corpus.join("factor_from_bytes"),
        corpus.join("block_factor_from_bytes"),
    );
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=case.rs");
    for dir in [&views, &factors] {
        // Watching the corpus is what makes a pruned or deleted one come back.
        println!("cargo::rerun-if-changed={}", dir.display());
        fs::create_dir_all(dir).expect("create the corpus directory");
    }
    for (name, input) in seeds() {
        let bytes = postcard::to_stdvec(&input).expect("serialize the seed");
        let decoded = postcard::from_bytes::<Input>(&bytes).expect("a seed decodes");
        // A rejected seed starts the fuzzer outside the factorization.
        let blocks =
            (decoded.run()).unwrap_or_else(|e| panic!("seed {name} reaches the solve: {e}"));
        assert!(blocks > 0, "seed {name} reaches no block");
        fs::write(views.join(format!("{name}.bin")), bytes).expect("write the seed");
        if let Input::F64(case) = decoded {
            let blocks = case.factor().expect("the seed factors").into_blocks();
            let bytes = postcard::to_stdvec(&blocks).expect("serialize the blocks");
            // A seed the validator rejects starts the fuzzer outside the solve.
            postcard::from_bytes::<Vec<BlockFactor>>(&bytes).expect("a real factor decodes");
            fs::write(factors.join(format!("{name}.bin")), bytes).expect("write the seed");
        }
    }
}

/// One seed per view and arm, so a mutation lands in a payload that already reaches the solve.
fn seeds() -> Vec<(&'static str, Input)> {
    let path = || {
        rows(
            &[0, 2, 5, 8, 10],
            &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            &[1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0],
        )
    };
    vec![
        (
            "csr_path_approximate",
            f64(View::Csr(path()), approximate()),
        ),
        ("csr_path_dense", f64(View::Csr(path()), dense())),
        (
            "csr_path_f32",
            Input::F32(case(View::Csr(narrowed(path())), approximate())),
        ),
        // Interleaved components relabel to a non-identity order.
        (
            "csr_permuted_components",
            f64(
                View::Csr(rows(
                    &[0, 2, 4, 6, 8],
                    &[0, 2, 1, 3, 0, 2, 1, 3],
                    &[2.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
                )),
                approximate(),
            ),
        ),
        (
            "csr_grounded_sddm",
            f64(
                View::Csr(rows(&[0, 2, 4], &[0, 1, 0, 1], &[2.0, -1.0, -1.0, 2.0])),
                approximate(),
            ),
        ),
        // Dominant only after scaling by diag(1, 2).
        (
            "csr_scaled",
            f64(
                View::Csr(rows(&[0, 2, 4], &[0, 1, 0, 1], &[4.0, -2.0, -2.0, 1.5])),
                approximate(),
            ),
        ),
        // An odd cycle of positive entries cannot fold to a Laplacian.
        (
            "csr_frustrated_triangle",
            f64(
                View::Csr(rows(
                    &[0, 3, 6, 9],
                    &[0, 1, 2, 0, 1, 2, 0, 1, 2],
                    &[2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
                )),
                approximate(),
            ),
        ),
        (
            "upper_csr_grid_4x4",
            f64(View::UpperCsr(upper_grid(4)), split(3)),
        ),
        (
            "gram_path",
            f64(
                View::Gram {
                    n: 4,
                    a: vec![0, 1, 2],
                    b: vec![1, 2, 3],
                    x: vec![1.0, 2.0, 1.0],
                    y: vec![-1.0, -2.0, -1.0],
                },
                approximate(),
            ),
        ),
        (
            "gram_mixed_signs",
            f64(
                View::Gram {
                    n: 3,
                    a: vec![0, 1, 0],
                    b: vec![1, 2, 2],
                    x: vec![1.0, 1.0, 1.0],
                    y: vec![1.0, -1.0, 2.0],
                },
                approximate(),
            ),
        ),
        (
            "bipartite_gram_slopes",
            f64(bipartite(true), exact_complement()),
        ),
        ("bipartite_gram_sampled_ac2", f64(bipartite(true), split(3))),
        (
            "bipartite_gram_incidence",
            f64(bipartite(false), approximate()),
        ),
        (
            "laplacian_grounded",
            f64(
                View::Laplacian {
                    n: 4,
                    a: vec![0, 1, 2],
                    b: vec![1, 2, 3],
                    weights: Weighting::SqrtOf(vec![1.0, -2.0, 0.5]),
                    ground: Some(vec![0.0, 0.0, 0.0, 0.5]),
                },
                approximate(),
            ),
        ),
        (
            "signed_graph_frustrated",
            f64(
                View::SignedGraph {
                    n: 3,
                    a: vec![0, 1, 0],
                    b: vec![1, 2, 2],
                    positive: vec![true, false, false],
                    weights: Weighting::Unit,
                },
                approximate(),
            ),
        ),
    ]
}

fn f64(view: View<f64>, build: Build) -> Input {
    Input::F64(case(view, build))
}

fn case<T>(view: View<T>, build: Build) -> Case<T> {
    Case {
        view,
        tolerance: 1e-9,
        max_iterations: 2048,
        build,
    }
}

fn rows(row_ptrs: &[u32], columns: &[u32], values: &[f64]) -> Rows<f64> {
    Rows {
        n: (row_ptrs.len() - 1) as u8,
        row_ptrs: row_ptrs.to_vec(),
        columns: columns.to_vec(),
        values: values.to_vec(),
    }
}

fn narrowed(r: Rows<f64>) -> Rows<f32> {
    Rows {
        n: r.n,
        row_ptrs: r.row_ptrs,
        columns: r.columns,
        values: r.values.iter().map(|&v| v as f32).collect(),
    }
}

/// The `k × k` grid Laplacian's upper triangle, enough steps for a mutation to reorder eliminations.
fn upper_grid(k: u32) -> Rows<f64> {
    let (mut row_ptrs, mut columns, mut values) = (vec![0], Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let degree = [i > 0, i + 1 < k, j > 0, j + 1 < k]
                .iter()
                .filter(|&&e| e)
                .count();
            columns.push(i * k + j);
            values.push(degree as f64);
            let right = (j + 1 < k).then_some(i * k + j + 1);
            let down = (i + 1 < k).then_some((i + 1) * k + j);
            for neighbour in [right, down].into_iter().flatten() {
                columns.push(neighbour);
                values.push(-1.0);
            }
            row_ptrs.push(columns.len() as u32);
        }
    }
    Rows {
        n: (k * k) as u8,
        row_ptrs,
        columns,
        values,
    }
}

/// Eight left rows on three right ones, two columns each; slopes on the left take the Gram path.
fn bipartite(slopes: bool) -> View<f64> {
    let columns = 16u32;
    let values = slopes.then(|| (0..columns).map(|k| 1.0 + f64::from(k % 5) / 4.0).collect());
    View::BipartiteGram {
        left: Side {
            n: 8,
            rows: (0..columns).map(|k| k / 2).collect(),
            values,
        },
        right: Side {
            n: 3,
            rows: (0..columns).map(|k| (k * 7 + k / 2) % 3).collect(),
            values: None,
        },
        weights: Weighting::Of((0..columns).map(|k| f64::from(1 + k % 3)).collect()),
    }
}

fn build(dense_up_to: u8) -> Build {
    Build {
        seed: 0,
        complement: Some(1),
        approximate: 1,
        dense_up_to,
    }
}

fn approximate() -> Build {
    build(0)
}

fn dense() -> Build {
    build(24)
}

fn split(copies: u8) -> Build {
    Build {
        complement: Some(copies),
        approximate: copies,
        ..approximate()
    }
}

fn exact_complement() -> Build {
    Build {
        complement: None,
        ..approximate()
    }
}
