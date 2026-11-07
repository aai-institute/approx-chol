use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(in crate::factor) trait Copies: Copy + Send + Sync {
    type Degrees: Degrees;

    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The copies kept of `copies` merged ones.
    fn cap(self, copies: u32) -> Self;

    /// Draws a partner position per copy; yields each distinct one once, with the copies that drew it and their summed `per_copy` weight.
    fn draw<'a, T: Real>(
        self,
        tally: &'a mut Vec<(usize, u32)>,
        per_copy: T,
        draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a;
}

/// Each vertex's copies over its incident edges, kept as edges come and go.
pub(in crate::factor) trait Degrees: Default {
    fn load(&mut self, n: usize);
    fn add(&mut self, v: usize, copies: u32);
    fn remove(&mut self, v: usize, copies: u32);
    fn release(&mut self, v: usize);
    fn get(&self, v: usize, len: u32) -> u64;
}

/// One copy per edge: a vertex's edge count is its degree.
#[derive(Default)]
pub(in crate::factor) struct EdgeCounts;

impl Degrees for EdgeCounts {
    fn load(&mut self, _: usize) {}

    fn add(&mut self, _: usize, _: u32) {}

    fn remove(&mut self, _: usize, _: u32) {}

    fn release(&mut self, _: usize) {}

    fn get(&self, _: usize, len: u32) -> u64 {
        u64::from(len)
    }
}

/// Summed per vertex: summing a hub's list per pivot would cost its degree.
#[derive(Default)]
pub(in crate::factor) struct CopySums(Vec<u64>);

impl Degrees for CopySums {
    fn load(&mut self, n: usize) {
        self.0.clear();
        self.0.resize(n, 0);
    }

    fn add(&mut self, v: usize, copies: u32) {
        self.0[v] += u64::from(copies);
    }

    fn remove(&mut self, v: usize, copies: u32) {
        self.0[v] -= u64::from(copies);
    }

    fn release(&mut self, v: usize) {
        self.0[v] = 0;
    }

    fn get(&self, v: usize, _: u32) -> u64 {
        self.0[v]
    }
}

#[derive(Clone, Copy)]
pub(in crate::factor) struct Single;

/// Virtual copy count: AC2's `k`, lowered by the merge cap, or the copies of one entry that drew one partner.
#[derive(Clone, Copy)]
pub(in crate::factor) struct Multi(u32);

impl Multi {
    /// AC2 at `k ≥ 2` copies; fewer is AC.
    pub(in crate::factor) fn split(k: u32) -> Option<Self> {
        (k >= 2).then_some(Self(k))
    }
}

impl Copies for Single {
    type Degrees = EdgeCounts;

    fn get(&self) -> u32 {
        1
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total
    }

    fn cap(self, _: u32) -> Self {
        Self
    }

    fn draw<'a, T: Real>(
        self,
        _: &'a mut Vec<(usize, u32)>,
        per_copy: T,
        mut draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a {
        std::iter::once((draw(), self, per_copy))
    }
}

impl Copies for Multi {
    type Degrees = CopySums;

    fn get(&self) -> u32 {
        self.0
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total / T::from(self.0).expect("a copy count is a float")
    }

    fn cap(self, copies: u32) -> Self {
        Self(copies.min(self.0))
    }

    fn draw<'a, T: Real>(
        self,
        drawn: &'a mut Vec<(usize, u32)>,
        per_copy: T,
        mut draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a {
        debug_assert!(self.0 > 0, "a capped entry keeps a copy");
        drawn.clear();
        drawn.push((draw(), 1));
        // Scanned, not sorted: sorting random draws mispredicts (bip_wide +26% cycles).
        for _ in 1..self.0 {
            let at = draw();
            match drawn.iter_mut().find(|(seen, _)| *seen == at) {
                Some((_, copies)) => *copies += 1,
                None => drawn.push((at, 1)),
            }
        }
        drawn.iter().map(move |&(at, copies)| {
            let weight = per_copy * T::from(copies).expect("a copy count is a float");
            (at, Self(copies), weight)
        })
    }
}
