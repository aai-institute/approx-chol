use std::mem::MaybeUninit;

use approx_chol::{ApplyError, MemStack, Real, StackReq};

/// A stack over `needed` and one slot more, every slot NaN: a slot read before it is written reaches the result.
pub fn nan_cells<T: Real>(needed: StackReq) -> Vec<MaybeUninit<T>> {
    nan_slots(needed.size_bytes() / size_of::<T>() + 1)
}

/// `slots` slots of `T`, every one NaN.
pub fn nan_slots<T: Real>(slots: usize) -> Vec<MaybeUninit<T>> {
    vec![MaybeUninit::new(T::nan()); slots]
}

/// The fewest slots of `T` `apply` accepts, at most `most`'s, by bisection: fewer are refused with `Scratch`.
pub fn least_slots<T: Real>(
    most: StackReq,
    mut apply: impl FnMut(&mut MemStack) -> Result<(), ApplyError>,
) -> usize {
    let mut cells = nan_cells::<T>(most);
    let mut accepts = |slots: usize| match apply(MemStack::new_any(&mut cells[..slots])) {
        Ok(()) => true,
        Err(ApplyError::Scratch) => false,
        Err(error) => panic!("only the stack may be short: {error}"),
    };
    let (mut low, mut high) = (0, most.size_bytes() / size_of::<T>());
    assert!(accepts(high), "`most` serves it");
    while low < high {
        let middle = (low + high) / 2;
        match accepts(middle) {
            true => high = middle,
            false => low = middle + 1,
        }
    }
    high
}
