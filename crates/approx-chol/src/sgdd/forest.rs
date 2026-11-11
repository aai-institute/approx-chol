use crate::Index;

/// What a link carries to its parent, composed along a path: nothing, a sign parity, or a ratio.
pub(super) trait Potential: Copy {
    /// A vertex's potential to itself.
    const ROOT: Self;
    fn then(self, up: Self) -> Self;
}

impl Potential for () {
    const ROOT: Self = ();
    fn then(self, _: Self) -> Self {}
}

impl Potential for bool {
    const ROOT: Self = false;
    fn then(self, up: Self) -> Self {
        self ^ up
    }
}

impl Potential for f64 {
    const ROOT: Self = 1.0;
    fn then(self, up: Self) -> Self {
        self * up
    }
}

pub(super) enum Under {
    A,
    B,
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

    /// Links two distinct roots by size, `a` on top at a tie; `under` gives the potential of the one going under.
    pub(super) fn link(
        &mut self,
        a: usize,
        b: usize,
        under: impl FnOnce(Under) -> P,
    ) -> (usize, P) {
        let (big, small, potential) = match self.size[a] >= self.size[b] {
            true => (a, b, under(Under::B)),
            false => (b, a, under(Under::A)),
        };
        self.links[small] = (I::from_index(big), potential);
        self.size[big] = self.size[big] + self.size[small];
        (big, potential)
    }
}

impl<I: Index> Forest<I, ()> {
    pub(super) fn union(&mut self, i: usize, j: usize) {
        let ((a, ()), (b, ())) = (self.find(i), self.find(j));
        if a != b {
            self.link(a, b, |_| ());
        }
    }
}
