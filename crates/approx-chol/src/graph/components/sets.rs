use super::layout::Layout;

/// Union-find with path halving and union by size.
pub(super) struct DisjointSets {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl DisjointSets {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
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

    /// Resolved in, surviving root out: one walk per caller, not one per edge.
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

    /// `None` when connected, which never pays for the counting sort below.
    pub(super) fn layout(&mut self) -> Option<Layout> {
        let total = self.parent.len();
        if total == 0 || self.is_one_set() {
            return None;
        }

        // Ascending, so components order by lowest member.
        let mut component_of = vec![u32::MAX; total];
        let mut ends: Vec<u32> = Vec::new();
        for vertex in 0..total {
            let root = self.find(vertex as u32) as usize;
            let component = component_of[root];
            if component == u32::MAX {
                component_of[root] = ends.len() as u32;
                ends.push(1);
            } else {
                ends[component as usize] += 1;
            }
        }

        // Exclusive scan: each entry is its component's cursor, which the fill advances.
        let mut start = 0u32;
        for count in &mut ends {
            let n = *count;
            *count = start;
            start += n;
        }
        let mut order = vec![0u32; total];
        let mut position = vec![0u32; total];
        for vertex in 0..total {
            let component = component_of[self.find(vertex as u32) as usize] as usize;
            order[ends[component] as usize] = vertex as u32;
            position[vertex] = ends[component];
            ends[component] += 1;
        }
        Some(Layout {
            order,
            position,
            ends,
        })
    }
}
