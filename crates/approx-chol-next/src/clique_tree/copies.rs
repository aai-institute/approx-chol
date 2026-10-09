use crate::config::SplitFactor;
use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(crate) trait Copies: Clone + Copy {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Factor: Copy;

    const SINGLE_COPY: bool;

    /// What a fill edge carries.
    fn one() -> Self;

    /// What an edge of the input carries once split.
    fn initial(factor: Self::Factor) -> Self;

    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, factor: Self::Factor) -> (Self, u32);
}

#[derive(Clone, Copy)]
pub(crate) struct Single;

/// Virtual copy count, only ever lowered from a [`SplitFactor`] by the merge cap.
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Copies for Single {
    type Factor = ();
    const SINGLE_COPY: bool = true;

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
    type Factor = SplitFactor;
    const SINGLE_COPY: bool = false;

    fn one() -> Self {
        Self(1)
    }

    fn initial(factor: SplitFactor) -> Self {
        Self(factor.get())
    }

    fn get(&self) -> u32 {
        self.0
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        todo!()
    }

    fn cap(copies: u32, factor: SplitFactor) -> (Self, u32) {
        todo!()
    }
}
