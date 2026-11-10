//! The envelope `(Header, B)` a persisted [`ApproxInverse`](super::ApproxInverse) is saved as.

use std::fmt;
use std::marker::PhantomData;

use serde::de::value::MapAccessDeserializer;
use serde::de::{Error, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::header::VersionMismatch;
use super::{Header, Scalar};
use crate::Real;

/// Accepts only [`Header::of::<T>()`](Header::of); in a sequence, another version fails before the scalar is read.
struct Checked<T>(PhantomData<T>);

impl<'de, T: Real> Deserialize<'de> for Checked<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_struct("Header", &["version", "scalar"], Self(PhantomData))
    }
}

impl<'de, T: Real> Visitor<'de> for Checked<T> {
    type Value = Self;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Header")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self, A::Error> {
        let expected = Header::of::<T>().version;
        let version: u32 = (seq.next_element()?).ok_or_else(|| Error::invalid_length(0, &self))?;
        if version != expected {
            return Err(Error::custom(VersionMismatch {
                found: version,
                expected,
            }));
        }
        let scalar: Scalar =
            (seq.next_element()?).ok_or_else(|| Error::invalid_length(1, &self))?;
        let found = Header { version, scalar };
        found.expect::<T>().map_err(Error::custom)?;
        Ok(self)
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self, A::Error> {
        let found = Header::deserialize(MapAccessDeserializer::new(map))?;
        found.expect::<T>().map_err(Error::custom)?;
        Ok(self)
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
