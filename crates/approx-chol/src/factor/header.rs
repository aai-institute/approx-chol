//! What a persisted [`ApproxInverse`](super::ApproxInverse) leads with: its format version and scalar type.

use std::fmt;

use crate::Real;

/// A persisted factor's format version and scalar type, read before the factor.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// Continues `approx-chol`'s `0x4143_xxxx`; it changes whenever a persisted factor's bytes do.
    pub version: u32,
    /// The scalar type its values are stored in.
    pub scalar: Scalar,
}

impl Header {
    /// The header this build writes for an `ApproxInverse<T>`, and the only one it reads.
    pub fn of<T: Real>() -> Header {
        Header {
            version: 0x4143_000E,
            scalar: T::SCALAR,
        }
    }

    /// Whether `self` is [`Header::of::<T>()`](Header::of).
    pub fn expect<T: Real>(self) -> Result<(), HeaderMismatch> {
        let expected = Header::of::<T>();
        match self == expected {
            true => Ok(()),
            false => Err(HeaderMismatch {
                found: self,
                expected,
            }),
        }
    }
}

/// The scalar type of a [`Real`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

impl Scalar {
    fn name(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }
}

/// A header other than the one this build reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderMismatch {
    /// The header read.
    pub found: Header,
    /// [`Header::of`] the requested scalar type.
    pub expected: Header,
}

impl fmt::Display for HeaderMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (found, expected) = (self.found, self.expected);
        let version = found.version != expected.version;
        if version {
            write!(
                f,
                "format version {:#010x}, expected format version {:#010x}",
                found.version, expected.version
            )?;
        }
        if found.scalar != expected.scalar {
            write!(
                f,
                "{}scalar {}, expected scalar {}",
                if version { "; " } else { "" },
                found.scalar.name(),
                expected.scalar.name()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for HeaderMismatch {}
