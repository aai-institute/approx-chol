use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(in crate::factor) trait Copies: Copy + Send + Sync {
    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, split: Self) -> (Self, u32);

    /// Draws a partner position per copy; yields each distinct one once, with the copies that drew it and their summed `per_copy` weight.
    fn draw<'a, T: Real>(
        self,
        tally: &'a mut Vec<(usize, u32)>,
        per_copy: T,
        draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a;
}

#[derive(Clone, Copy)]
pub(in crate::factor) struct Single;

/// Virtual copy count: `Multiplicity::Split`'s `k`, lowered by the merge cap, or the copies of one entry that drew one partner.
#[derive(Clone, Copy)]
pub(in crate::factor) struct Multi(pub(in crate::factor) u32);

impl Copies for Single {
    fn get(&self) -> u32 {
        1
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total
    }

    fn cap(copies: u32, Single: Self) -> (Self, u32) {
        (Self, copies - 1)
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
    fn get(&self) -> u32 {
        self.0
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total / T::from(self.0).expect("a copy count is a float")
    }

    fn cap(copies: u32, Self(split): Self) -> (Self, u32) {
        let kept = copies.min(split);
        (Self(kept), copies - kept)
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
