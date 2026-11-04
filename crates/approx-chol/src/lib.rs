//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one irreducible block at a time.
//!
//! ```
//! use approx_chol::{factorize, CsrRef};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let x = factorize(csr)?.solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs, unreachable_pub)]

/// Serde through `remote = "Self"`, every decoded value passed through its type's `checked`.
#[cfg(feature = "serde")]
macro_rules! checked_serde {
    ($ty:ident) => {
        impl<T: serde::Serialize, I: crate::storage::Index + serde::Serialize> serde::Serialize
            for $ty<T, I>
        {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                $ty::serialize(self, serializer)
            }
        }
        impl<'de, T, I> serde::Deserialize<'de> for $ty<T, I>
        where
            T: crate::storage::Real + serde::Deserialize<'de>,
            I: crate::storage::Index + serde::Deserialize<'de>,
        {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let corrupt = || serde::de::Error::custom(concat!("corrupt ", stringify!($ty)));
                let value = $ty::deserialize(deserializer)?;
                value.checked().ok_or_else(corrupt)
            }
        }
    };
}

mod error;
mod factor;
pub mod input;
mod sgdd;
mod storage;

use num_traits::PrimInt;

pub use error::Error;
use factor::Storage;
#[cfg(feature = "serde")]
pub use factor::FORMAT_VERSION as FACTOR_FORMAT_VERSION;
pub use factor::{BlockFactor, Config, Factor, InnerParallelism, SchurMode, SolveError, Vertices};
pub use input::{CsrError, CsrRef, IndexKind, Input, OwnedCsr, Side};
use sgdd::{Complement, Reduced};
pub use sgdd::{ReduceError, ScalingPolicy, ScalingReport};
pub use storage::Real;

/// Factorizes any [`Input`] with [`Config::default`] and [`ScalingPolicy::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<Factor<T>, Error>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: TryInto<Input<'a, T, I>>,
    <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
{
    factorize_with(input, Config::default(), ScalingPolicy::default())
}

/// Factorizes any [`Input`]: [`Sgdd::reduce`] under `scaling`, then [`Sgdd::factor`] under `config`.
pub fn factorize_with<'a, T, I, M>(
    input: M,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<Factor<T>, Error>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: TryInto<Input<'a, T, I>>,
    <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
{
    Ok(Sgdd::reduce(input, scaling)?.factor(&config))
}

/// Factors a bipartite `A`'s Schur complement onto its `kept` side, formed under [`Config::complement`] and factored as [`factorize_with`] would; the factor's vertex `k` is the kept side's `k`-th.
pub fn factorize_complement<T: Real, I: PrimInt + Sync>(
    input: input::BipartiteFactorWidth2<'_, T, I>,
    kept: Side,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<Factor<T>, Error> {
    Ok(match fits_u32(input.n()) {
        true => Factor::build(
            Complement::reduce_onto(input, kept, scaling)?,
            &config,
            Storage::U32,
        ),
        false => Factor::build(
            Complement::reduce_onto(input, kept, scaling)?,
            &config,
            Storage::U64,
        ),
    })
}

/// Whether `u32` indexes a cover's two sheets and its ground, the most slots any block of `n` vertices takes.
fn fits_u32(n: usize) -> bool {
    u32::try_from(n.saturating_mul(2).saturating_add(1)).is_ok()
}

/// Built only by [`Sgdd::reduce`]: `A` in symmetric generalized diagonally dominant form, each irreducible block scaled to dominance.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64>(Indexed<T>);

/// The reduction in the narrowest index its cover's `2n + 1` slots fit.
#[derive(Clone, Debug)]
enum Indexed<T> {
    U32(Reduced<T, u32>),
    U64(Reduced<T, u64>),
}

impl<T: Real> Sgdd<T> {
    /// Reduces any [`Input`], scaling each block short of dominance under `scaling`; nothing after this reads `input`.
    pub fn reduce<'a, I, M>(input: M, scaling: ScalingPolicy) -> Result<Self, Error>
    where
        I: PrimInt + Sync + 'a,
        M: TryInto<Input<'a, T, I>>,
        <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
    {
        let input = input.try_into().map_err(Into::into)?;
        Ok(Self(match fits_u32(input.n()) {
            true => Indexed::U32(Reduced::reduce(input, scaling)?),
            false => Indexed::U64(Reduced::reduce(input, scaling)?),
        }))
    }

    /// Factors each block in parallel; clone first to factor again under another `config`.
    pub fn factor(self, config: &Config) -> Factor<T> {
        match self.0 {
            Indexed::U32(summands) => Factor::build(summands, config, Storage::U32),
            Indexed::U64(summands) => Factor::build(summands, config, Storage::U64),
        }
    }
}

/// `units` per task so each carries `2^bits` of `work`, in `u64` as a 32-bit `usize` wraps past `2^(32-bits)` units.
fn per_task(units: usize, work: u64, bits: u32) -> usize {
    let per = ((units as u64) << bits).div_ceil(work.max(1));
    usize::try_from(per).unwrap_or(usize::MAX).max(1)
}
