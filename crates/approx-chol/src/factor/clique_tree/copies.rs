use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(in crate::factor) trait Copies: Copy + Send + Sync {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Split: Copy + Send + Sync;

    /// Scratch [`Copies::draw`] reuses across entries: nothing for one copy.
    type Tally: Default + Send;

    /// What an edge of the input carries once split.
    fn initial(split: Self::Split) -> Self;

    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, split: Self::Split) -> (Self, u32);

    /// Draws a partner position per copy; yields each distinct one once, with the copies that drew it and their summed `per_copy` weight.
    fn draw<'a, T: Real>(
        self,
        tally: &'a mut Self::Tally,
        per_copy: T,
        draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a;
}

#[derive(Clone, Copy)]
pub(in crate::factor) struct Single;

/// Virtual copy count: `Multiplicity::Split`'s `k`, lowered by the merge cap, or the copies of one entry that drew one partner.
#[derive(Clone, Copy)]
pub(in crate::factor) struct Multi(u32);

impl Copies for Single {
    type Split = ();
    type Tally = ();

    fn initial((): ()) -> Self {
        Self
    }

    fn get(&self) -> u32 {
        1
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total
    }

    fn cap(copies: u32, (): ()) -> (Self, u32) {
        (Self, copies - 1)
    }

    fn draw<'a, T: Real>(
        self,
        (): &'a mut (),
        per_copy: T,
        mut draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a {
        std::iter::once((draw(), self, per_copy))
    }
}

impl Copies for Multi {
    type Split = u32;
    type Tally = Vec<usize>;

    fn initial(split: u32) -> Self {
        Self(split)
    }

    fn get(&self) -> u32 {
        self.0
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        total / T::from(self.0).expect("a copy count is a float")
    }

    fn cap(copies: u32, split: u32) -> (Self, u32) {
        let kept = copies.min(split);
        (Self(kept), copies - kept)
    }

    fn draw<'a, T: Real>(
        self,
        drawn: &'a mut Vec<usize>,
        per_copy: T,
        draw: impl FnMut() -> usize,
    ) -> impl Iterator<Item = (usize, Self, T)> + 'a {
        drawn.clear();
        drawn.extend(std::iter::repeat_with(draw).take(self.0 as usize));
        drawn.sort_unstable();
        drawn.chunk_by(|a, b| a == b).map(move |run| {
            let copies = Self(run.len() as u32);
            let weight = per_copy * T::from(copies.0).expect("a copy count is a float");
            (run[0], copies, weight)
        })
    }
}
