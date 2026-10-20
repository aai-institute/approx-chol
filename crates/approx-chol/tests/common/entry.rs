use approx_chol::input::{
    BipartiteGram, CsrRef, FactorWidth1, FactorWidth2, Laplacian, SignedGraph, UpperCsrRef, Weights,
};
use approx_chol::{factorize_with, Config, Factor, Real};
use num_traits::{NumCast, PrimInt};

use crate::graph::Graph;

/// The input entry points; each reads one [`Graph`] where it [`states`](Entry::states) it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    Csr,
    UpperCsr,
    Gram,
    BipartiteGram,
    Laplacian,
    GroundedLaplacian,
    SignedGraph,
}

/// The scalar a factor computes in and the index type its input view states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Types {
    F32U32,
    F32U64,
    F64U32,
    F64U64,
}

impl Types {
    pub fn epsilon(self) -> f64 {
        match self {
            Self::F32U32 | Self::F32U64 => f32::EPSILON.into(),
            Self::F64U32 | Self::F64U64 => f64::EPSILON,
        }
    }
}

/// One value per [`Types`], so a property written once in `f64` runs on all four.
pub enum ByType<A, B, C, D> {
    F32U32(A),
    F32U64(B),
    F64U32(C),
    F64U64(D),
}

macro_rules! by_type {
    ($value:expr, $x:ident => $body:expr) => {
        match $value {
            $crate::entry::ByType::F32U32($x) => $body,
            $crate::entry::ByType::F32U64($x) => $body,
            $crate::entry::ByType::F64U32($x) => $body,
            $crate::entry::ByType::F64U64($x) => $body,
        }
    };
}
pub(crate) use by_type;

pub type Factors = ByType<Factor<f32>, Factor<f32>, Factor<f64>, Factor<f64>>;

/// `labels.0[v]` is caller vertex `v`'s index in the entry point's numbering.
#[derive(Clone, Debug)]
pub struct Labels(Vec<usize>);

impl Labels {
    pub fn to_local(&self, b: &[f64]) -> Vec<f64> {
        let mut local = vec![0.0; b.len()];
        (self.0.iter().zip(b)).for_each(|(&at, &value)| local[at] = value);
        local
    }

    pub fn to_caller(&self, x: &[f64]) -> Vec<f64> {
        self.0.iter().map(|&at| x[at]).collect()
    }

    fn graph(&self, graph: &Graph) -> Graph {
        let mut relabeled = graph.clone();
        for edge in &mut relabeled.edges {
            (edge.a, edge.b) = (self.0[edge.a], self.0[edge.b]);
        }
        relabeled.ground = self.to_local(&graph.ground);
        relabeled
    }
}

/// A two-colouring numbered left side first, as a bipartite Gram takes it; `None` on an odd cycle.
fn bipartition(graph: &Graph) -> Option<(Labels, usize)> {
    let n = graph.n;
    let mut adjacency = vec![Vec::new(); n];
    for edge in &graph.edges {
        adjacency[edge.a].push(edge.b);
        adjacency[edge.b].push(edge.a);
    }
    let (mut right, mut root_of) = (vec![None; n], vec![0; n]);
    for root in 0..n {
        if right[root].is_some() {
            continue;
        }
        right[root] = Some(false);
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            root_of[v] = root;
            let side = right[v];
            for &u in &adjacency[v] {
                match right[u] {
                    None => {
                        right[u] = side.map(|side| !side);
                        stack.push(u);
                    }
                    other if other == side => return None,
                    _ => {}
                }
            }
        }
    }
    let mut right: Vec<bool> = right.into_iter().map(|side| side == Some(true)).collect();
    // A ground's column needs a row on the other side; only isolated vertices can all be left.
    if n >= 2 && !right.contains(&true) {
        let last = root_of[n - 1];
        (0..n)
            .filter(|&v| root_of[v] == last)
            .for_each(|v| right[v] = true);
    }
    let n_left = right.iter().filter(|&&r| !r).count();
    let (mut lefts, mut rights) = (0..n_left, n_left..n);
    let labels = right.iter().map(|&r| match r {
        false => lefts.next(),
        true => rights.next(),
    });
    Some((Labels(labels.map(Option::unwrap).collect()), n_left))
}

pub fn cast<S: NumCast, D: NumCast>(x: S) -> D {
    D::from(x).expect("representable")
}

pub fn cast_all<S: NumCast + Copy, D: NumCast>(xs: &[S]) -> Vec<D> {
    xs.iter().map(|&x| cast(x)).collect()
}

impl Entry {
    /// A Gram needs a second vertex beside a ground; a Laplacian view states no positive edge.
    pub fn states(self, graph: &Graph) -> bool {
        let signed = graph.edges.iter().any(|edge| edge.positive);
        let grounded = graph.ground.iter().any(|&g| g != 0.0);
        let lone_ground = graph.n < 2 && grounded;
        match self {
            Entry::Csr | Entry::UpperCsr => true,
            Entry::Gram => !lone_ground,
            Entry::BipartiteGram => !lone_ground && bipartition(graph).is_some(),
            Entry::Laplacian => !signed && !grounded,
            Entry::GroundedLaplacian => !signed,
            Entry::SignedGraph => !grounded,
        }
    }

    pub fn reduce(self, graph: &Graph, types: Types) -> Reduced {
        assert!(self.states(graph), "{self:?} cannot state {graph:?}");
        let (labels, n_left) = match self {
            Entry::BipartiteGram => bipartition(graph).expect("states a bipartite graph"),
            _ => (Labels((0..graph.n).collect()), 0),
        };
        Reduced {
            entry: self,
            local: labels.graph(graph),
            n_left,
            types,
            labels,
        }
    }

    fn factorize_as<T: Real, I: PrimInt + Sync>(
        self,
        graph: &Graph,
        n_left: usize,
        config: Config,
    ) -> Factor<T> {
        let n = graph.n;
        let rejected = |error: &dyn std::fmt::Debug| -> Factor<T> {
            panic!("{self:?} rejected {graph:?}: {error:?}")
        };
        let (a, b): (Vec<I>, Vec<I>) = (graph.edges.iter())
            .map(|edge| (cast::<usize, I>(edge.a), cast::<usize, I>(edge.b)))
            .unzip();
        let weights: Vec<T> = graph.edges.iter().map(|edge| cast(edge.weight)).collect();
        let weights = Weights::of(&weights).expect("nonnegative weights");
        match self {
            Entry::Csr | Entry::UpperCsr => {
                let upper = self == Entry::UpperCsr;
                let (ptrs, columns, values) = graph.csr(upper);
                let (ptrs, columns): (Vec<I>, Vec<I>) = (cast_all(&ptrs), cast_all(&columns));
                let values: Vec<T> = cast_all(&values);
                match upper {
                    false => {
                        let csr =
                            CsrRef::new(&ptrs, &columns, &values, n as u32).expect("valid CSR");
                        factorize_with(csr, config)
                    }
                    true => {
                        let csr = UpperCsrRef::new(&ptrs, &columns, &values, n as u32)
                            .expect("valid CSR");
                        factorize_with(csr, config)
                    }
                }
                .unwrap_or_else(|error| rejected(&error))
            }
            Entry::Gram => {
                let (mut a, mut b, mut x, mut y) = (a, b, vec![], vec![]);
                for edge in &graph.edges {
                    let root = edge.weight.sqrt();
                    x.push(cast::<f64, T>(root));
                    y.push(cast(edge.sign() * root));
                }
                for (i, &g) in graph.ground.iter().enumerate().filter(|(_, &g)| g != 0.0) {
                    // A zero partner leaves the column one nonzero: the ground.
                    a.push(cast(i));
                    b.push(cast((i + 1) % n));
                    x.push(cast(g.sqrt()));
                    y.push(T::zero());
                }
                let v =
                    FactorWidth2::new(n, &a, &b, &x, &y, Weights::unit()).expect("valid factor");
                factorize_with(v, config).unwrap_or_else(|error| rejected(&error))
            }
            Entry::BipartiteGram => {
                let n_right = n - n_left;
                let (mut left, mut right) = (Vec::<I>::new(), Vec::<I>::new());
                let (mut x, mut y, mut w) = (vec![], vec![], vec![]);
                for edge in &graph.edges {
                    let (l, r) = match edge.a < n_left {
                        true => (edge.a, edge.b),
                        false => (edge.b, edge.a),
                    };
                    left.push(cast(l));
                    right.push(cast(r - n_left));
                    x.push(1.0);
                    y.push(edge.sign());
                    w.push(edge.weight);
                }
                for (i, &g) in graph.ground.iter().enumerate().filter(|(_, &g)| g != 0.0) {
                    // A zero on the other side leaves the column one nonzero: the ground.
                    let (l, r, x_i, y_i) = match i < n_left {
                        true => (i, 0, 1.0, 0.0),
                        false => (0, i - n_left, 0.0, 1.0),
                    };
                    left.push(cast(l));
                    right.push(cast(r));
                    x.push(x_i);
                    y.push(y_i);
                    w.push(g);
                }
                // Incidence on a side is `None`, so an unvalued Gram takes its own path.
                let valued = |values: &[f64]| values.iter().any(|&v| v != 1.0);
                let (x_t, y_t, w_t): (Vec<T>, Vec<T>, Vec<T>) =
                    (cast_all(&x), cast_all(&y), cast_all(&w));
                let left = FactorWidth1::new(n_left, &left, valued(&x).then_some(&x_t[..]));
                let right = FactorWidth1::new(n_right, &right, valued(&y).then_some(&y_t[..]));
                let weights = Weights::of(&w_t).expect("nonnegative weights");
                let gram = BipartiteGram::new(
                    left.expect("valid left factor"),
                    right.expect("valid right factor"),
                    weights,
                )
                .expect("valid bipartite Gram");
                factorize_with(gram, config).unwrap_or_else(|error| rejected(&error))
            }
            Entry::Laplacian | Entry::GroundedLaplacian => {
                let view = Laplacian::new(n, &a, &b, weights).expect("valid Laplacian");
                let ground: Vec<T> = cast_all(&graph.ground);
                match self {
                    Entry::GroundedLaplacian => {
                        factorize_with(view.grounded(&ground).expect("valid ground"), config)
                    }
                    _ => factorize_with(view, config),
                }
                .unwrap_or_else(|error| rejected(&error))
            }
            Entry::SignedGraph => {
                let positive: Vec<bool> = graph.edges.iter().map(|edge| edge.positive).collect();
                let view =
                    SignedGraph::new(n, &a, &b, &positive, weights).expect("valid signed graph");
                factorize_with(view, config).unwrap_or_else(|error| rejected(&error))
            }
        }
    }
}

/// A graph as its entry point states it, in that entry point's numbering.
pub struct Reduced {
    entry: Entry,
    local: Graph,
    n_left: usize,
    types: Types,
    labels: Labels,
}

impl Reduced {
    pub fn factorize(&self, config: &Config) -> Factored {
        let (entry, graph, n_left, config) = (self.entry, &self.local, self.n_left, *config);
        let factor = match self.types {
            Types::F32U32 => ByType::F32U32(entry.factorize_as::<f32, u32>(graph, n_left, config)),
            Types::F32U64 => ByType::F32U64(entry.factorize_as::<f32, u64>(graph, n_left, config)),
            Types::F64U32 => ByType::F64U32(entry.factorize_as::<f64, u32>(graph, n_left, config)),
            Types::F64U64 => ByType::F64U64(entry.factorize_as::<f64, u64>(graph, n_left, config)),
        };
        Factored {
            factor,
            labels: self.labels.clone(),
        }
    }
}

/// A factor that solves in the caller's numbering and in `f64`.
pub struct Factored {
    pub factor: Factors,
    pub labels: Labels,
}

impl Factored {
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let local = self.labels.to_local(b);
        let x = crate::entry::by_type!(&self.factor, factor => solve(factor, &local));
        self.labels.to_caller(&x)
    }
}

fn solve<T: Real>(factor: &Factor<T>, b: &[f64]) -> Vec<f64> {
    let x = factor
        .solve(&cast_all::<f64, T>(b))
        .expect("b has length n");
    cast_all(&x)
}
