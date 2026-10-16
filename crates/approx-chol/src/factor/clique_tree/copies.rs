use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(in crate::factor) trait Copies: Copy + Send + Sync {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Split: Copy + Send + Sync;

    /// What a fill edge carries.
    fn one() -> Self;

    /// What an edge of the input carries once split.
    fn initial(split: Self::Split) -> Self;

    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, split: Self::Split) -> (Self, u32);
}

#[derive(Clone, Copy)]
pub(in crate::factor) struct Single;

/// Virtual copy count, only ever lowered from `Multiplicity::Split`'s `k` by the merge cap.
#[derive(Clone, Copy)]
pub(in crate::factor) struct Multi(u32);

impl Copies for Single {
    type Split = ();
    fn one() -> Self {
        Self
    }

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
}

impl Copies for Multi {
    type Split = u32;
    fn one() -> Self {
        Self(1)
    }

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
}
