use std::mem::MaybeUninit;

use approx_chol::{Real, StackReq};

/// A stack over `needed` and one slot more, every slot NaN: a slot read before it is written reaches the result.
pub fn nan_cells<T: Real>(needed: StackReq) -> Vec<MaybeUninit<T>> {
    vec![MaybeUninit::new(T::nan()); needed.size_bytes() / size_of::<T>() + 1]
}
