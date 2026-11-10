//! The envelope `(Header, B)` a persisted [`ApproxInverse`](super::ApproxInverse) is saved as.

use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::Header;
use crate::Real;

/// Accepts only [`Header::of::<T>()`](Header::of).
struct Checked<T>(PhantomData<T>);

impl<'de, T: Real> Deserialize<'de> for Checked<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let found = Header::deserialize(deserializer)?;
        found.expect::<T>().map_err(serde::de::Error::custom)?;
        Ok(Self(PhantomData))
    }
}

/// `(Header, body)`: a pair, so every format reads the header before the body.
pub(crate) fn serialize<T: Real, S: Serializer>(
    serializer: S,
    body: impl Serialize,
) -> Result<S::Ok, S::Error> {
    (Header::of::<T>(), body).serialize(serializer)
}

/// The body behind [`Header::of::<T>()`](Header::of); any other header fails before the body is read.
pub(crate) fn deserialize<'de, T: Real, D: Deserializer<'de>, B: Deserialize<'de>>(
    deserializer: D,
) -> Result<B, D::Error> {
    let (Checked::<T>(_), body) = <(Checked<T>, B)>::deserialize(deserializer)?;
    Ok(body)
}
