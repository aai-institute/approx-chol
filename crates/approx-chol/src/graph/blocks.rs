/// Every vertex once, components back to back. One array rather than one per
/// component: the same sequence answers all three questions asked of it.
pub(crate) struct BlockLayout {
    pub(super) order: Vec<u32>,
    /// The next block starts where this one stops, so no block claims a vertex twice
    /// or leaves a gap.
    pub(super) ends: Vec<u32>,
}

impl BlockLayout {
    pub(crate) fn block_count(&self) -> usize {
        self.ends.len()
    }

    /// Each block's global vertex names, in storage order.
    pub(crate) fn blocks(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.ends.iter().scan(0usize, |start, &end| {
            let end = end as usize;
            let block = &self.order[*start..end];
            *start = end;
            Some(block)
        })
    }

    /// The same sequence read as a permutation.
    pub(crate) fn into_order(self) -> Vec<u32> {
        self.order
    }
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
    pub(super) fn local(&self, global: usize) -> usize {
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
    pub(super) fn last(&self) -> u32 {
        match self {
            Self::Whole(n) => (n - 1) as u32,
            Self::Part { vertices, .. } => {
                *vertices.last().expect("a block has at least one vertex")
            }
        }
    }
}
