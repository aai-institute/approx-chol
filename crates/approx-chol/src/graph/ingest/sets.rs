use super::blocks::BlockLayout;

/// Union-find with path halving and union by size.
pub(super) struct DisjointSets {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSets {
    /// Room for the ground vertex, which [`push`](Self::push) appends after the walk:
    /// sizing exactly would make augmented input pay two reallocations for it.
    pub(super) fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n + 1);
        parent.extend(0..n as u32);
        let mut size = Vec::with_capacity(n + 1);
        size.resize(n, 1);
        Self { parent, size }
    }

    /// Appends a vertex the CSR does not carry, and names it.
    pub(super) fn push(&mut self) -> u32 {
        let vertex = self.parent.len() as u32;
        self.parent.push(vertex);
        self.size.push(1);
        vertex
    }

    pub(super) fn find(&mut self, mut vertex: u32) -> u32 {
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
    pub(super) fn union_resolved(&mut self, root: u32, vertex: u32) -> u32 {
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
    pub(super) fn layout(&mut self) -> Option<BlockLayout> {
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
