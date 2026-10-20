//! The format version a persisted [`BlockFactor`](crate::Block) leads with.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Continues `approx-chol`'s `0x4143_xxxx`; bump it whenever a persisted factor's bytes change.
pub const FORMAT_VERSION: u32 = 0x4143_0009;

/// Writes [`FORMAT_VERSION`] and decodes only it.
struct FormatVersion;

impl Serialize for FormatVersion {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(FORMAT_VERSION)
    }
}

impl<'de> Deserialize<'de> for FormatVersion {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let found = u32::deserialize(deserializer)?;
        match found == FORMAT_VERSION {
            true => Ok(Self),
            false => Err(serde::de::Error::custom(format_args!(
                "format version {found:#010x}, expected format version {FORMAT_VERSION:#010x}"
            ))),
        }
    }
}

/// `(FORMAT_VERSION, body)`: a pair, so every format reads the version before the body.
pub(crate) fn serialize<S: Serializer>(
    serializer: S,
    body: impl Serialize,
) -> Result<S::Ok, S::Error> {
    (FormatVersion, body).serialize(serializer)
}

/// The body behind [`FORMAT_VERSION`]; any other version fails before the body is read.
pub(crate) fn deserialize<'de, D: Deserializer<'de>, B: Deserialize<'de>>(
    deserializer: D,
) -> Result<B, D::Error> {
    let (FormatVersion, body) = <(FormatVersion, B)>::deserialize(deserializer)?;
    Ok(body)
}
