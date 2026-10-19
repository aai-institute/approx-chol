use crate::Real;

/// An edge's virtual copies; AC is AC2 at one copy per edge, so this trait is the whole difference.
pub(in crate::factor) trait Copies: Copy + Send + Sync {
    /// `Single` cannot name a `k`, making AC over split multi-edges a type error.
    type Split: Copy + Send + Sync;

    /// What [`Copies::draw`] keeps between entries: nothing for one copy.
    type Tally: Default + Send;

    /// What an edge of the input carries once split.
    fn initial(split: Self::Split) -> Self;

    fn get(&self) -> u32;
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// The kept copies, and how many the cap discarded.
    fn cap(copies: u32, split: Self::Split) -> (Self, u32);

    /// Draws a partner position per copy and emits each distinct one once, with the copies that drew it and their summed `per_copy` weight.
    fn draw<T: Real>(
        self,
        tally: &mut Self::Tally,
        per_copy: T,
        draw: impl FnMut() -> usize,
        emit: impl FnMut(usize, Self, T),
    );
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

    fn draw<T: Real>(
        self,
        (): &mut (),
        per_copy: T,
        mut draw: impl FnMut() -> usize,
        mut emit: impl FnMut(usize, Self, T),
    ) {
        emit(draw(), self, per_copy);
    }
}

/// Per star position, how many of one entry's copies drew it, zero between entries; and the positions drawn, in first-draw order.
#[derive(Default)]
pub(in crate::factor) struct Tally {
    counts: Vec<u32>,
    drawn: Vec<usize>,
}

impl Copies for Multi {
    type Split = u32;
    type Tally = Tally;

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

    fn draw<T: Real>(
        self,
        tally: &mut Tally,
        per_copy: T,
        mut draw: impl FnMut() -> usize,
        mut emit: impl FnMut(usize, Self, T),
    ) {
        for _ in 0..self.0 {
            let position = draw();
            if position >= tally.counts.len() {
                tally.counts.resize(position + 1, 0);
            }
            if tally.counts[position] == 0 {
                tally.drawn.push(position);
            }
            tally.counts[position] += 1;
        }
        for position in tally.drawn.drain(..) {
            let copies = std::mem::take(&mut tally.counts[position]);
            let total = per_copy * T::from(copies).expect("a copy count is a float");
            emit(position, Self(copies), total);
        }
    }
}
