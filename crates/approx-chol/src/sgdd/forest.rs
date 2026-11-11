use crate::Index;

/// What a link carries to its parent, composed along a path.
pub(super) trait Potential: Copy {
    /// A vertex's potential to itself.
    const ROOT: Self;
    /// What a link from `b` up to `a` carries before it is stored.
    type Link: Copy + Into<Self>;
    fn then(self, up: Self) -> Self;
    fn reversed(link: Self::Link) -> Self::Link;
}

impl Potential for () {
    const ROOT: Self = ();
    type Link = Self;
    fn then(self, _: Self) -> Self {}
    fn reversed(link: Self) -> Self {
        link
    }
}

/// Union by size with path halving, each vertex holding its potential to its parent.
pub(super) struct Forest<I, P> {
    /// Parent and potential to it, together: for `()` the layout and bounds checks of a bare parent array.
    links: Vec<(I, P)>,
    size: Vec<I>,
}

impl<I: Index, P: Potential> Forest<I, P> {
    pub(super) fn new(n: usize) -> Self {
        Self {
            links: (0..n).map(|i| (I::from_index(i), P::ROOT)).collect(),
            size: vec![I::one(); n],
        }
    }

    pub(super) fn len(&self) -> usize {
        self.links.len()
    }

    pub(super) fn size(&self, root: usize) -> I {
        self.size[root]
    }

    /// Root and potential to it; every visited vertex skips to its grandparent.
    pub(super) fn find(&mut self, mut i: usize) -> (usize, P) {
        let mut to_root = P::ROOT;
        while self.links[i].0.index() != i {
            let (parent, potential) = self.links[i];
            let (grandparent, up) = self.links[parent.index()];
            let halved = potential.then(up);
            self.links[i] = (grandparent, halved);
            to_root = to_root.then(halved);
            i = grandparent.index();
        }
        (i, to_root)
    }

    /// Links two distinct roots by size, `a` on top at a tie: `b` goes under `a` by `b_to_a`, else `a` under `b` by its reverse.
    pub(super) fn link(&mut self, a: usize, b: usize, b_to_a: P::Link) -> (usize, P) {
        let (big, small, link) = match self.size[a] >= self.size[b] {
            true => (a, b, b_to_a),
            false => (b, a, P::reversed(b_to_a)),
        };
        let potential = link.into();
        self.links[small] = (I::from_index(big), potential);
        self.size[big] = self.size[big] + self.size[small];
        (big, potential)
    }
}

impl<I: Index> Forest<I, ()> {
    pub(super) fn union(&mut self, i: usize, j: usize) {
        let ((a, ()), (b, ())) = (self.find(i), self.find(j));
        if a != b {
            self.link(a, b, ());
        }
    }
}
