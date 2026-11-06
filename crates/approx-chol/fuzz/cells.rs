use std::mem::MaybeUninit;

use approx_chol::StackReq;

/// A stack's cells over `needed`, every one `fill`: a slot read before it is written reaches the solution.
pub fn filled<T: Copy>(needed: StackReq, fill: T) -> Vec<MaybeUninit<T>> {
    vec![MaybeUninit::new(fill); needed.size_bytes() / size_of::<T>()]
}
