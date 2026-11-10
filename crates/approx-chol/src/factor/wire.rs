//! The envelope `(Header, B)` a persisted [`ApproxInverse`](super::ApproxInverse) is saved as.

use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::Header;
use crate::Real;

/// Writes [`Header::of::<T>()`](Header::of) and decodes only it.
struct Current<T>(PhantomData<T>);

impl<T: Real> Serialize for Current<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Header::of::<T>().serialize(serializer)
    }
}

impl<'de, T: Real> Deserialize<'de> for Current<T> {
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
    (Current::<T>(PhantomData), body).serialize(serializer)
}

/// The body behind [`Header::of::<T>()`](Header::of); any other header fails before the body is read.
pub(crate) fn deserialize<'de, T: Real, D: Deserializer<'de>, B: Deserialize<'de>>(
    deserializer: D,
) -> Result<B, D::Error> {
    let (Current::<T>(_), body) = <(Current<T>, B)>::deserialize(deserializer)?;
    Ok(body)
}
