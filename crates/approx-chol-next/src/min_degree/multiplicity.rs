use crate::config::SplitFactor;
use crate::Real;

/// AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(super) trait EdgeCount: Clone + Copy {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Split: Copy;

    const SINGLE_COPY: bool;

    fn one() -> Self;
    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, limit: Self::Split) -> (Self, u32);
}

#[derive(Clone, Copy)]
pub(super) struct Single;

/// Virtual copy count, only ever lowered from a [`SplitFactor`] by the merge cap.
#[derive(Clone, Copy)]
pub(super) struct Multi(u32);

impl EdgeCount for Single {
    type Split = ();
    const SINGLE_COPY: bool = true;

    fn one() -> Self {
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

impl EdgeCount for Multi {
    type Split = SplitFactor;
    const SINGLE_COPY: bool = false;

    fn one() -> Self {
        Self(1)
    }

    fn get(&self) -> u32 {
        self.0
    }

    fn per_copy<T: Real>(&self, total: T) -> T {
        todo!()
    }

    fn cap(copies: u32, limit: SplitFactor) -> (Self, u32) {
        todo!()
    }
}
