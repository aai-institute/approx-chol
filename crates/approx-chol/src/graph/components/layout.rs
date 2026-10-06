/// Every vertex once, components back to back, in one array since one sequence answers every query.
pub(super) struct Layout {
    pub(super) order: Vec<u32>,
    /// `order` inverted, so a component's local index is a subtraction, written once for all.
    pub(super) position: Vec<u32>,
    /// The next component starts where this one stops, so none overlaps or leaves a gap.
    pub(super) ends: Vec<u32>,
}

impl Layout {
    /// Each component's vertices, in storage order.
    pub(super) fn components(&self) -> impl Iterator<Item = Vertices<'_>> + '_ {
        self.ends.iter().scan(0u32, |start, &end| {
            let part = Vertices::Part {
                vertices: &self.order[*start as usize..end as usize],
                position: &self.position,
                start: *start,
            };
            *start = end;
            Some(part)
        })
    }
}

/// One component's vertices and the map back; [`Whole`](Vertices::Whole) never materializes `0..n`.
pub(super) enum Vertices<'v> {
    Whole(usize),
    Part {
        vertices: &'v [u32],
        /// Every input vertex's place in the layout, shared by all components.
        position: &'v [u32],
        start: u32,
    },
}

impl Vertices<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Whole(n) => *n,
            Self::Part { vertices, .. } => vertices.len(),
        }
    }

    #[inline]
    pub(super) fn global(&self, local: usize) -> usize {
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

    pub(super) fn first(&self) -> u64 {
        match self {
            Self::Whole(_) => 0,
            Self::Part { vertices, .. } => u64::from(vertices[0]),
        }
    }
}
