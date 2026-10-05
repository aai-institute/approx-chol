/// Every vertex once, components back to back. One array rather than one per
/// component: the same sequence answers all three questions asked of it.
pub(crate) struct BlockLayout {
    pub(super) order: Vec<u32>,
    /// `order` inverted, so a component's local index is a subtraction, written once for all.
    pub(super) position: Vec<u32>,
    /// The next block starts where this one stops, so no block claims a vertex twice
    /// or leaves a gap.
    pub(super) ends: Vec<u32>,
}

impl BlockLayout {
    pub(crate) fn block_count(&self) -> usize {
        self.ends.len()
    }

    /// Each block's vertices, in storage order.
    pub(crate) fn blocks(&self) -> impl Iterator<Item = BlockVertices<'_>> + '_ {
        self.ends.iter().scan(0u32, |start, &end| {
            let block = BlockVertices::Part {
                vertices: &self.order[*start as usize..end as usize],
                position: &self.position,
                start: *start,
            };
            *start = end;
            Some(block)
        })
    }

    /// The same sequence read as a permutation.
    pub(crate) fn into_order(self) -> Vec<u32> {
        self.order
    }
}

/// One block's vertices and the map back. [`Whole`](BlockVertices::Whole) is the
/// connected case, which never materializes `0..n`.
pub(crate) enum BlockVertices<'v> {
    Whole(usize),
    Part {
        vertices: &'v [u32],
        /// Every input vertex's place in the layout, shared by all blocks.
        position: &'v [u32],
        start: u32,
    },
}

impl BlockVertices<'_> {
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
            Self::Part {
                position, start, ..
            } => (position[global] - start) as usize,
        }
    }

    /// Names the block by what it holds rather than by how many blocks precede it.
    pub(crate) fn first(&self) -> u64 {
        match self {
            Self::Whole(_) => 0,
            Self::Part { vertices, .. } => u64::from(vertices[0]),
        }
    }
}
