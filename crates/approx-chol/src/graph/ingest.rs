//! CSR to elimination graph: canonicalize, pair each off-diagonal with its mirror,
//! and close the row deficits with a Gremban ground vertex.

use super::{add_edge_pair, AdjListGraph, BlockLayout, Edge, EdgeCount};
use crate::types::{count_as_scalar, Real};
use crate::{CsrError, CsrRef, Error};

/// Union-find with path halving and union by size.
struct DisjointSets {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSets {
    /// Room for the ground vertex, which [`push`](Self::push) appends after the walk:
    /// sizing exactly would make augmented input pay two reallocations for it.
    fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n + 1);
        parent.extend(0..n as u32);
        let mut size = Vec::with_capacity(n + 1);
        size.resize(n, 1);
        Self { parent, size }
    }

    /// Appends a vertex the CSR does not carry, and names it.
    fn push(&mut self) -> u32 {
        let vertex = self.parent.len() as u32;
        self.parent.push(vertex);
        self.size.push(1);
        vertex
    }

    fn find(&mut self, mut vertex: u32) -> u32 {
        while self.parent[vertex as usize] != vertex {
            let grandparent = self.parent[self.parent[vertex as usize] as usize];
            self.parent[vertex as usize] = grandparent;
            vertex = grandparent;
        }
        vertex
    }

    /// Whether every vertex is already in one set, which is the connected case.
    fn is_one_set(&mut self) -> bool {
        let root = self.find(0);
        self.size[root as usize] as usize == self.parent.len()
    }

    /// Takes `root` already resolved and hands back the surviving root, so a caller
    /// unioning one vertex against many pays a single walk for it rather than one per
    /// edge.
    fn union_resolved(&mut self, root: u32, vertex: u32) -> u32 {
        let (mut root, mut merged) = (root, self.find(vertex));
        if root == merged {
            return root;
        }
        if self.size[root as usize] < self.size[merged as usize] {
            core::mem::swap(&mut root, &mut merged);
        }
        self.parent[merged as usize] = root;
        self.size[root as usize] += self.size[merged as usize];
        root
    }

    /// The blocks these unions imply, so each one is known — and routed — before any
    /// graph is built for it. `None` when the graph is connected, which is the one
    /// block case and never pays for the counting sort below.
    fn layout(&mut self) -> Option<BlockLayout> {
        let total = self.parent.len();
        if total == 0 || self.is_one_set() {
            return None;
        }

        // Ascending, so a block's vertices are appended in order and the blocks
        // themselves are ordered by their lowest member — the ground vertex outranks
        // every real one, so it lands last in its own block.
        let mut block_of = vec![u32::MAX; total];
        let mut ends: Vec<u32> = Vec::new();
        for vertex in 0..total {
            let root = self.find(vertex as u32) as usize;
            let block = block_of[root];
            if block == u32::MAX {
                block_of[root] = ends.len() as u32;
                ends.push(1);
            } else {
                ends[block as usize] += 1;
            }
        }

        // Exclusive scan, so each entry is its block's write cursor; the fill below then
        // advances every cursor to exactly the end it is named for.
        let mut start = 0u32;
        for count in &mut ends {
            let n = *count;
            *count = start;
            start += n;
        }
        let mut order = vec![0u32; total];
        for vertex in 0..total {
            let block = block_of[self.find(vertex as u32) as usize] as usize;
            order[ends[block] as usize] = vertex as u32;
            ends[block] += 1;
        }
        Some(BlockLayout { order, ends })
    }
}

/// The input read as ingestion needs it: strictly ascending columns within every row.
/// The caller's arrays already are that whenever scipy produced them, which is the only
/// shape within emits, so the rewritten copy is what rare input costs rather than what
/// every input pays.
struct Canonical<'a, T> {
    input: CsrRef<'a, T, u32>,
    /// `None` when the caller's arrays are already canonical.
    rewritten: Option<Rewritten<T>>,
}

impl<'a, T: Real> Canonical<'a, T> {
    /// Canonical input never reads a value here: [`validate`] visits every stored entry
    /// exactly once, so checking each as it is read spares ingestion a whole stream over
    /// the widest of the three arrays.
    fn of(csr: CsrRef<'a, T, u32>) -> Result<Self, Error> {
        if is_canonical(csr.row_ptrs(), csr.col_indices()) {
            return Ok(Self {
                input: csr,
                rewritten: None,
            });
        }
        // Rows tile the value array in order, so this position is the caller's own.
        // Scanning before rewriting is what keeps it so, and is why non-finite is
        // reported in preference to the non-canonical shape.
        if let Some(position) = csr.values().iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { position });
        }
        Ok(Self {
            input: csr,
            rewritten: Some(rewrite(csr)?),
        })
    }

    fn arrays(&self) -> (&[u32], &[u32], &[T]) {
        match &self.rewritten {
            Some(r) => (&r.row_ptrs, &r.col_indices, &r.values),
            None => (
                self.input.row_ptrs(),
                self.input.col_indices(),
                self.input.values(),
            ),
        }
    }

    /// Additions charged to a row's excess, counted on the caller's arrays before
    /// coalescing: [`rewrite`] folds each duplicate group with its own additions, and
    /// those land in the row's sum too. For canonical input this is the diagonal plus
    /// the row's degree.
    fn terms(&self) -> impl Iterator<Item = u32> + '_ {
        self.input
            .row_ptrs()
            .windows(2)
            .map(|bounds| bounds[1] - bounds[0])
    }
}

/// Accumulated rather than short-circuited: canonical input is the overwhelming case and
/// runs every entry regardless, so the early exit only adds a branch per entry to the
/// path that always takes it. Measured worth 0.4-2% of the build against the `all` form.
fn is_canonical(row_ptrs: &[u32], col_indices: &[u32]) -> bool {
    let mut canonical = true;
    for bounds in row_ptrs.windows(2) {
        let row = &col_indices[bounds[0] as usize..bounds[1] as usize];
        for pair in row.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    canonical
}

/// Canonical arrays rebuilt from non-canonical input.
struct Rewritten<T> {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    values: Vec<T>,
}

/// Only non-canonical input pays this copy.
fn rewrite<T: Real>(csr: CsrRef<'_, T, u32>) -> Result<Rewritten<T>, Error> {
    let nnz = csr.col_indices().len();
    let mut row_ptrs = Vec::with_capacity(csr.n() + 1);
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(u32, T)> = Vec::new();
    row_ptrs.push(0u32);
    for (row, (cols, vals)) in csr.rows().enumerate() {
        entries.clear();
        entries.extend(cols.iter().copied().zip(vals.iter().copied()));
        // One row's degree, not nnz. Stable, so duplicates sum in stored order.
        entries.sort_by_key(|&(col, _)| col);
        for group in entries.chunk_by(|left, right| left.0 == right.0) {
            let folded = group[1..].iter().fold(group[0].1, |sum, &(_, v)| sum + v);
            // Every input was finite, so only this fold can overflow. Caught here so the
            // arrays handed on are all-finite, which is what lets the reads downstream
            // report a position in the caller's numbering.
            if !folded.is_finite() {
                return Err(Error::NonFiniteRow { row });
            }
            col_indices.push(group[0].0);
            values.push(folded);
        }
        row_ptrs.push(col_indices.len() as u32);
    }
    Ok(Rewritten {
        row_ptrs,
        col_indices,
        values,
    })
}

/// One forward-only cursor per row. The walk is a merge-join only because every entry
/// is claimed at most once, which is what [`Canonical`] guarantees.
struct Mirrors<'a, T> {
    row_ptrs: &'a [u32],
    col_indices: &'a [u32],
    values: &'a [T],
    cursors: Vec<u32>,
}

impl<'a, T: Real> Mirrors<'a, T> {
    fn new(row_ptrs: &'a [u32], col_indices: &'a [u32], values: &'a [T]) -> Self {
        let cursors = row_ptrs[..row_ptrs.len() - 1].to_vec();
        Self {
            row_ptrs,
            col_indices,
            values,
            cursors,
        }
    }

    /// Stored zeros count as absent.
    fn claim(&mut self, row: usize, col: usize) -> Result<T, Error> {
        let row_end = self.row_ptrs[row + 1];
        let mut cursor = self.cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = self.col_indices[cursor as usize] as usize;
            if at > col {
                break;
            }
            let value = self.values[cursor as usize];
            if !value.is_finite() {
                return Err(Error::NonFiniteValue {
                    position: cursor as usize,
                });
            }
            cursor += 1;
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                self.cursors[row] = cursor;
                return Err(Error::Asymmetric { edge: (at, row) });
            }
        }
        self.cursors[row] = cursor;
        Ok(found)
    }
}

fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

/// Everything ingestion learns from the CSR before any graph exists.
struct Ingested<T> {
    /// One per real vertex, and the ground vertex's last when grounded.
    diagonal: Vec<T>,
    grounding: Grounding<T>,
    /// Blocks implied by the connectivity the validating walk unioned as it went, rather
    /// than by a pass of its own: every edge is already being visited there.
    layout: Option<BlockLayout>,
}

/// A matrix whose rows all balance is a bare Laplacian on each of its components, and
/// has no ground vertex to attach.
enum Grounding<T> {
    Floating,
    Grounded {
        /// Indexed by row, zero where the row balances — the row-sum accumulator reused
        /// in place rather than a second array.
        surpluses: Vec<T>,
        /// How many of those are positive, which is the ground vertex's degree.
        degree: usize,
    },
}

/// One pass per row, claiming each upper-triangle entry's mirror. Every stored entry is
/// read exactly once between the claims and the loop below, which is what lets
/// [`Canonical::of`] leave finiteness to this walk. Accumulates what the
/// augmentation decision needs and builds nothing: which arm each block gets is not
/// known until its dimension is, and that waits on the layout this pass feeds.
fn validate<T: Real>(canonical: &Canonical<'_, T>) -> Result<Ingested<T>, Error> {
    let (row_ptrs, col_indices, values) = canonical.arrays();
    let n = row_ptrs.len() - 1;
    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);

    let mut sets = DisjointSets::new(n);
    let mut diagonal = vec![T::zero(); n];
    // Off-diagonal contributions only; the diagonal joins in `ground`, which is
    // not known for a row until that row's own claim below.
    let mut row_sums = vec![T::zero(); n];

    for row in 0..n {
        let row_end = row_ptrs[row + 1];
        // The diagonal is claimed like any mirror, which both yields its value and
        // advances the cursor past everything below it — by now only the zeros that
        // contribute no edge. Claiming every diagonal up front instead would skip
        // the mirrors that live below them.
        diagonal[row] = mirrors.claim(row, row)?;
        let mut cursor = mirrors.cursors[row];
        let mut root = sets.find(row as u32);

        while cursor < row_end {
            let col = col_indices[cursor as usize] as usize;
            let upper = values[cursor as usize];
            if !upper.is_finite() {
                return Err(Error::NonFiniteValue {
                    position: cursor as usize,
                });
            }
            cursor += 1;
            // Duplicates can coalesce to exactly zero, which contributes no edge.
            if upper == T::zero() {
                continue;
            }
            let lower = mirrors.claim(col, row)?;
            if !approximately_equal(upper, lower) {
                return Err(Error::Asymmetric { edge: (row, col) });
            }
            if upper > T::zero() {
                return Err(Error::PositiveOffDiagonal { edge: (row, col) });
            }
            // Zero was skipped and positive rejected, so `upper` is negative here
            // and `-upper` is the edge weight. A NaN coalesced from opposing
            // infinities never reaches this line — it fails the symmetry check.
            // Each row sums the value it stores, not the one the graph symmetrizes to:
            // charging `upper` to both would make the tolerated mirror difference look
            // like `col`'s own surplus and ground it.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + lower;
            root = sets.union_resolved(root, col as u32);
        }
    }
    ground(diagonal, row_sums, canonical.terms(), sets)
}

/// How far one row's diagonal exceeds its off-diagonal mass, judged against the noise
/// the row's own scale and term count can carry.
enum RowBalance<T> {
    NonFinite,
    Deficit,
    Negligible,
    /// Worth closing with a ground edge.
    Surplus(T),
}

impl<T: Real> RowBalance<T> {
    /// `terms` is how many additions produced `excess`, not the row's degree.
    fn of(diagonal: T, off_diagonal_sum: T, terms: u32) -> Self {
        let excess = diagonal + off_diagonal_sum;
        // Every off-diagonal folded into the sum was negative, so the row's magnitude
        // sum is `|d| + d - excess` and needs no second accumulator. Subtracting before
        // the second add keeps a diagonal above `MAX / 2` from overflowing to infinity.
        let scale = (diagonal.abs() - excess) + diagonal;
        // A non-finite sum forces a non-finite scale, so scale alone decides. Every
        // comparison below succeeds on an infinite deficit (`-inf < -inf`).
        if !scale.is_finite() {
            return Self::NonFinite;
        }
        // The most this row's own additions could have invented, and so the only
        // departure from zero-sum the row cannot account for. One floor for both signs:
        // a departure this clears is real evidence whichever way it points, and
        // forgiving more in one direction than the other grounds a row for a drift that
        // would be dismissed as noise with its sign flipped.
        let accumulated = T::epsilon() * scale * count_as_scalar::<T, _>(terms);
        if excess < -accumulated {
            return Self::Deficit;
        }
        if excess <= accumulated {
            return Self::Negligible;
        }
        Self::Surplus(excess)
    }
}

/// `row_sums` arrives holding each row's off-diagonal total; the diagonal joins it
/// here, the first point at which every row's is known.
fn ground<T: Real>(
    mut diagonal: Vec<T>,
    mut row_sums: Vec<T>,
    terms: impl Iterator<Item = u32>,
    mut sets: DisjointSets,
) -> Result<Ingested<T>, Error> {
    let mut total = T::zero();
    let mut degree = 0usize;
    for (row, ((sum, &d), count)) in row_sums
        .iter_mut()
        .zip(diagonal.iter())
        .zip(terms)
        .enumerate()
    {
        *sum = match RowBalance::of(d, *sum, count) {
            RowBalance::NonFinite => return Err(Error::NonFiniteRow { row }),
            RowBalance::Deficit => return Err(Error::NotDiagonallyDominant { row }),
            RowBalance::Negligible => T::zero(),
            RowBalance::Surplus(excess) => {
                total = total + excess;
                degree += 1;
                excess
            }
        };
    }

    let m = diagonal.len();
    if degree == 0 {
        return Ok(Ingested {
            diagonal,
            grounding: Grounding::Floating,
            layout: sets.layout(),
        });
    }
    if m >= u32::MAX as usize {
        return Err(Error::InvalidCsr(
            CsrError::MatrixDimensionExceedsIndexType {
                n: m.saturating_add(1),
            },
        ));
    }
    diagonal.push(total);
    // The ground vertex is absent from the CSR, so the rows it closes are unioned
    // through it here: it links every component holding a surplus into one block, and
    // connectivity read from the CSR alone would hand back each of them separately.
    let vertex = sets.push();
    let mut root = vertex;
    for (row, &surplus) in row_sums.iter().enumerate() {
        if surplus > T::zero() {
            root = sets.union_resolved(root, row as u32);
        }
    }
    Ok(Ingested {
        diagonal,
        grounding: Grounding::Grounded {
            surpluses: row_sums,
            degree,
        },
        layout: sets.layout(),
    })
}

/// One block's vertices and the map back from global names to its own.
///
/// [`Whole`](BlockVertices::Whole) is the connected case, where a local vertex already
/// is a global one, so the common input never materializes `0..n` to say so.
pub(crate) enum BlockVertices<'v> {
    Whole(usize),
    Part {
        vertices: &'v [u32],
        /// Only the entries `vertices` names are meaningful; the rest belong to other
        /// blocks and are never read through this view.
        local_of: &'v [u32],
    },
}

impl<'v> BlockVertices<'v> {
    pub(crate) fn whole(n: usize) -> Self {
        Self::Whole(n)
    }

    /// Fills `local_of`, so "the reverse map agrees with `vertices`" is established here
    /// once instead of being a precondition every reader has to be trusted to have met.
    pub(crate) fn part(vertices: &'v [u32], local_of: &'v mut [u32]) -> Self {
        for (local, &global) in vertices.iter().enumerate() {
            local_of[global as usize] = local as u32;
        }
        Self::Part { vertices, local_of }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Whole(n) => *n,
            Self::Part { vertices, .. } => vertices.len(),
        }
    }

    #[inline]
    pub(crate) fn global(&self, local: usize) -> usize {
        match self {
            Self::Whole(_) => local,
            Self::Part { vertices, .. } => vertices[local] as usize,
        }
    }

    #[inline]
    fn local(&self, global: usize) -> usize {
        match self {
            Self::Whole(_) => global,
            Self::Part { local_of, .. } => local_of[global] as usize,
        }
    }

    /// Names the block by what it holds rather than by how many blocks precede it.
    pub(crate) fn first(&self) -> u64 {
        match self {
            Self::Whole(_) => 0,
            Self::Part { vertices, .. } => u64::from(vertices[0]),
        }
    }

    /// Blocks list their vertices ascending, so the highest-numbered one is last.
    fn last(&self) -> u32 {
        match self {
            Self::Whole(n) => (n - 1) as u32,
            Self::Part { vertices, .. } => {
                *vertices.last().expect("a block has at least one vertex")
            }
        }
    }
}

/// The ingested input, kept whole so each block takes only what its backend needs. A
/// block routed to the dense arm reads its triangle straight from these arrays; only
/// one that will actually be eliminated on gets an adjacency list built for it.
pub(crate) struct Ingestion<'a, T> {
    canonical: Canonical<'a, T>,
    /// Emptied by [`take_block_diagonal`](Ingestion::take_block_diagonal) for the whole
    /// graph, which is why `n` is its own field rather than this vector's length.
    diagonal: Vec<T>,
    n: usize,
    grounding: Grounding<T>,
    layout: Option<BlockLayout>,
}

impl<'a, T: Real> Ingestion<'a, T> {
    pub(crate) fn of(csr: CsrRef<'a, T, u32>) -> Result<Self, Error> {
        let canonical = Canonical::of(csr)?;
        let Ingested {
            diagonal,
            grounding,
            layout,
        } = validate(&canonical)?;
        Ok(Self {
            canonical,
            n: diagonal.len(),
            diagonal,
            grounding,
            layout,
        })
    }

    /// Vertices the factorization covers, the ground one included.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Whether this block holds the Gremban ground vertex, which decides how it is
    /// anchored. Ingestion appends that vertex above every real one and a block lists
    /// its vertices ascending, so it can only ever be a block's last.
    pub(crate) fn carries_ground(&self, block: &BlockVertices<'_>) -> bool {
        match &self.grounding {
            Grounding::Floating => false,
            // The bound checked in `ground` is what makes the cast lossless.
            Grounding::Grounded { surpluses, .. } => block.last() == surpluses.len() as u32,
        }
    }

    /// `None` when the graph is connected, which is the one block case. Taken rather
    /// than borrowed so the caller can walk the blocks while asking for each one.
    pub(crate) fn take_layout(&mut self) -> Option<BlockLayout> {
        self.layout.take()
    }

    /// The diagonal entry the block's row `local` carries.
    pub(crate) fn block_diagonal(&self, block: &BlockVertices<'_>, local: usize) -> T {
        self.diagonal[block.global(local)]
    }

    /// Hands `entry` each strictly-upper off-diagonal of the block's row `local`, in
    /// the block's own numbering.
    ///
    /// The upper mirror is the authoritative one: `validate` tolerates mirrors that
    /// differ by a few ulps, and the approximate route symmetrizes on this same value,
    /// so reading the stored lower one instead would make the two routes disagree about
    /// a matrix they were both handed.
    ///
    /// A block's last vertex has no CSR row when it is the ground vertex, and no upper
    /// entries when it is not, so it yields nothing either way rather than indexing past
    /// the row pointers.
    pub(crate) fn upper_row(
        &self,
        block: &BlockVertices<'_>,
        local: usize,
        mut entry: impl FnMut(usize, T),
    ) {
        let (row_ptrs, col_indices, values) = self.canonical.arrays();
        let row = block.global(local);
        if row + 1 >= row_ptrs.len() {
            return;
        }
        let (from, to) = (row_ptrs[row] as usize, row_ptrs[row + 1] as usize);
        for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
            if col as usize > row && value != T::zero() {
                entry(block.local(col as usize), value);
            }
        }
    }

    /// The block's diagonal in local order. The whole graph is moved out rather than
    /// copied: it is one block, so nothing reads it again.
    pub(crate) fn take_block_diagonal(&mut self, block: &BlockVertices<'_>) -> Vec<T> {
        match block {
            BlockVertices::Whole(_) => core::mem::take(&mut self.diagonal),
            BlockVertices::Part { vertices, .. } => vertices
                .iter()
                .map(|&vertex| self.diagonal[vertex as usize])
                .collect(),
        }
    }

    /// Builds the block's adjacency, which only the approximate arm needs.
    pub(crate) fn block_graph<C: EdgeCount>(
        &self,
        block: &BlockVertices<'_>,
    ) -> AdjListGraph<C, T> {
        let (row_ptrs, col_indices, values) = self.canonical.arrays();
        let rows = row_ptrs.len() - 1;
        let n = block.len();
        // Every grounded row is unioned through the ground vertex, so they all share one
        // block and its degree is the whole count wherever it lands.
        let ground_degree = match self.grounding {
            Grounding::Floating => 0,
            Grounding::Grounded { degree, .. } => degree,
        };

        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(n);
        for local in 0..n {
            let global = block.global(local);
            let degree = if global < rows {
                (row_ptrs[global + 1] - row_ptrs[global]) as usize
            } else {
                ground_degree
            };
            adj.push(Vec::with_capacity(degree));
        }

        // The arm is resolved once around the edge loop, not tested inside it: measured
        // on this loop, the per-edge discriminant test kept `local_of`'s pointer and
        // length spilled to the stack and reloaded for every edge added.
        match block {
            BlockVertices::Whole(_) => {
                for local in 0..n {
                    if local >= rows {
                        continue;
                    }
                    let (from, to) = (row_ptrs[local] as usize, row_ptrs[local + 1] as usize);
                    for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
                        if col as usize > local && value != T::zero() {
                            add_edge_pair(&mut adj, local, col as usize, -value);
                        }
                    }
                }
            }
            BlockVertices::Part { vertices, local_of } => {
                // Columns are already bounded by the matrix dimension, so narrowing the
                // slice lets the bound live in a register rather than being reloaded.
                let local_of = &local_of[..rows];
                for (local, &global) in vertices.iter().enumerate() {
                    let global = global as usize;
                    if global >= rows {
                        continue;
                    }
                    let (from, to) = (row_ptrs[global] as usize, row_ptrs[global + 1] as usize);
                    for (&col, &value) in col_indices[from..to].iter().zip(&values[from..to]) {
                        if col as usize > global && value != T::zero() {
                            add_edge_pair(&mut adj, local, local_of[col as usize] as usize, -value);
                        }
                    }
                }
            }
        }

        if let Grounding::Grounded { surpluses, .. } = &self.grounding {
            if self.carries_ground(block) {
                let ground = n - 1;
                for (row, &surplus) in surpluses.iter().enumerate() {
                    // The clamp in `ground` left every surplus non-negative.
                    if surplus > T::zero() {
                        add_edge_pair(&mut adj, block.local(row), ground, surplus);
                    }
                }
            }
        }

        AdjListGraph::from_adjacency(adj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Single;

    /// Blocks are what the layout says they are, in its own numbering.
    fn blocks_of(row_ptrs: &[u32], col_indices: &[u32], values: &[f64]) -> Option<Vec<Vec<u32>>> {
        let n = (row_ptrs.len() - 1) as u32;
        let csr = CsrRef::new(row_ptrs, col_indices, values, n).expect("valid CSR");
        let canonical = Canonical::of(csr).expect("canonical");
        validate(&canonical)
            .expect("valid SDDM")
            .layout
            .map(|layout| layout.blocks().map(<[u32]>::to_vec).collect::<Vec<_>>())
    }

    /// Two PD blocks whose off-diagonal graphs are disjoint. The ground vertex is not
    /// in the CSR, so connectivity read from the CSR alone splits them — and each half
    /// then gets an adjacency built for a block that does not hold all of its edges.
    #[test]
    fn components_sharing_a_ground_vertex_are_one_block() {
        let blocks = blocks_of(
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[5.0, -1.0, -1.0, 4.0, 5.0, -1.0, -1.0, 4.0],
        );
        assert!(
            blocks.is_none(),
            "a shared ground vertex makes the augmented graph connected, got {blocks:?}"
        );
    }

    /// The same shape with no surplus anywhere: nothing grounds them, so they stay
    /// apart. Without this the test above would also pass on a layout that merged
    /// every component unconditionally.
    #[test]
    fn components_with_no_surplus_stay_separate() {
        let blocks = blocks_of(
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
        );
        assert_eq!(blocks, Some(vec![vec![0, 1], vec![2, 3]]));
    }

    /// One grounded component beside a floating one. The ground vertex outranks every
    /// real vertex, so it lands last in the block it joins rather than opening one.
    #[test]
    fn the_ground_vertex_lands_last_in_its_own_block() {
        let blocks = blocks_of(
            &[0, 2, 4, 6, 8],
            &[0, 1, 0, 1, 2, 3, 2, 3],
            &[5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0],
        );
        assert_eq!(blocks, Some(vec![vec![0, 1, 4], vec![2, 3]]));
    }

    /// A vertex no edge reaches is its own block, which is what makes the ordering
    /// "by lowest member" observable rather than incidental.
    #[test]
    fn blocks_are_ordered_by_their_lowest_vertex() {
        let blocks = blocks_of(
            &[0, 2, 3, 4, 6],
            &[0, 3, 1, 2, 0, 3],
            &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0],
        );
        assert_eq!(blocks, Some(vec![vec![0, 3], vec![1], vec![2]]));
    }

    /// The layout is read before any graph exists, so this pins that the graph a block
    /// is actually handed still has the vertices the layout promised it.
    #[test]
    fn the_built_graph_agrees_with_the_layout() {
        let row_ptrs = [0u32, 2, 4, 6, 8];
        let col_indices = [0u32, 1, 0, 1, 2, 3, 2, 3];
        let values = [5.0, -1.0, -1.0, 4.0, 1.0, -1.0, -1.0, 1.0];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4).expect("valid CSR");
        let mut ingestion = Ingestion::of(csr).expect("valid SDDM");
        assert_eq!(ingestion.n(), 5);

        let layout = ingestion.take_layout().expect("two blocks");
        let blocks: Vec<Vec<u32>> = layout.blocks().map(<[u32]>::to_vec).collect();
        assert_eq!(blocks, vec![vec![0, 1, 4], vec![2, 3]]);

        let mut local_of = vec![0u32; ingestion.n()];
        let built: Vec<(usize, bool)> = blocks
            .iter()
            .map(|vertices| {
                let view = BlockVertices::part(vertices, &mut local_of);
                let carries = ingestion.carries_ground(&view);
                (ingestion.block_graph::<Single>(&view).n(), carries)
            })
            .collect();
        assert_eq!(
            built,
            vec![(3, true), (2, false)],
            "only the block ending at the ground vertex is anchored to it"
        );
    }
}
