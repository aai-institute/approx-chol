//! The format version a persisted [`ApproxInverse`](super::ApproxInverse) leads with.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::marker::PhantomData;

use super::order::{Order, Slots};
use super::{Block, BlockElimination, Cholesky, Isolated, Storage};
use crate::sgdd::{Congruence, Selection};
use crate::{Index, Real};

/// Continues `approx-chol`'s `0x4143_xxxx`; bump it whenever a persisted factor's bytes change.
pub const FORMAT_VERSION: u32 = 0x4143_000A;

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

/// Every block in order, each under its storage tag with its vertices beside its parts, as the factor's blocks were stored before it owned their vertices.
pub(super) struct Blocks<'a, T>(pub(super) &'a [Storage<T>], pub(super) &'a Order);

impl<T: Serialize> Serialize for Blocks<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let blocks = self.0.iter().zip(self.1.blocks());
        serializer.collect_seq(blocks.map(|(block, slots)| StoredRef::of(block, slots)))
    }
}

#[derive(Serialize)]
#[serde(rename = "Storage", bound(serialize = "T: Serialize"))]
enum StoredRef<'a, T> {
    U32(BlockRef<'a, T, u32>),
    U64(BlockRef<'a, T, u64>),
    Isolated(IsolatedRef<'a, T>),
}

impl<'a, T> StoredRef<'a, T> {
    fn of(block: &'a Storage<T>, slots: Slots<'a>) -> Self {
        match block {
            Storage::U32(block) => Self::U32(BlockRef::of(block, slots)),
            Storage::U64(block) => Self::U64(BlockRef::of(block, slots)),
            Storage::Isolated(isolated) => Self::Isolated(IsolatedRef {
                vertex: slots.vertex(0) as u64,
                inverse_pivot: &isolated.inverse_pivot,
            }),
        }
    }
}

#[derive(Serialize)]
#[serde(
    rename = "Block",
    bound(serialize = "T: Serialize, I: Index + Serialize")
)]
struct BlockRef<'a, T, I> {
    restriction: RestrictionRef<'a, T, I>,
    elimination: &'a Option<Box<BlockElimination<T, I>>>,
    cholesky: &'a Cholesky<T, I>,
    grounding_unproven: bool,
}

impl<'a, T, I> BlockRef<'a, T, I> {
    fn of(block: &'a Block<T, I>, slots: Slots<'a>) -> Self {
        Self {
            restriction: RestrictionRef {
                vertices: Vertices::new(slots),
                congruence: &block.congruence,
            },
            elimination: &block.elimination,
            cholesky: &block.cholesky,
            grounding_unproven: block.grounding_unproven,
        }
    }
}

#[derive(Serialize)]
#[serde(
    rename = "Restriction",
    bound(serialize = "T: Serialize, I: Index + Serialize")
)]
struct RestrictionRef<'a, T, I> {
    vertices: Vertices<'a, I>,
    congruence: &'a Congruence<T>,
}

/// The vertex list either way, each as an `I`, so the wire never shows a run.
pub(super) struct Vertices<'a, I>(Slots<'a>, PhantomData<I>);

impl<'a, I> Vertices<'a, I> {
    pub(super) fn new(slots: Slots<'a>) -> Self {
        Self(slots, PhantomData)
    }
}

impl<I: Index + Serialize> Serialize for Vertices<'_, I> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.vertices().map(I::from_index))
    }
}

#[derive(Serialize)]
#[serde(rename = "Isolated", bound(serialize = "T: Serialize"))]
struct IsolatedRef<'a, T> {
    vertex: u64,
    inverse_pivot: &'a T,
}

/// `(n, dead, blocks, clamps)` as decoded.
pub(super) type Body<T> = (usize, Box<[usize]>, Decoded<T>, Box<[crate::Clamp]>);

/// The decoded blocks, the coupled ones apart from the isolated vertices, each in the bytes' order.
#[derive(Deserialize)]
#[serde(
    from = "Vec<Stored<T>>",
    bound(deserialize = "T: Real + Deserialize<'de>")
)]
pub(super) struct Decoded<T> {
    pub(super) coupled: Vec<Coupled<T>>,
    pub(super) lone: Vec<IsolatedWire<T>>,
}

impl<T> From<Vec<Stored<T>>> for Decoded<T> {
    fn from(stored: Vec<Stored<T>>) -> Self {
        let (mut coupled, mut lone) = (Vec::new(), Vec::new());
        for block in stored {
            match block {
                Stored::U32(block) => coupled.push(Coupled::U32(block)),
                Stored::U64(block) => coupled.push(Coupled::U64(block)),
                Stored::Isolated(isolated) => lone.push(isolated),
            }
        }
        Self { coupled, lone }
    }
}

#[derive(Deserialize)]
#[serde(rename = "Storage", bound(deserialize = "T: Real + Deserialize<'de>"))]
enum Stored<T> {
    U32(BlockWire<T, u32>),
    U64(BlockWire<T, u64>),
    Isolated(IsolatedWire<T>),
}

/// A coupled block as the bytes hold it, its vertices not yet the factor's.
pub(super) enum Coupled<T> {
    U32(BlockWire<T, u32>),
    U64(BlockWire<T, u64>),
}

impl<T: Real> Coupled<T> {
    pub(super) fn slots(&self) -> Slots<'_> {
        match self {
            Self::U32(block) => Slots::of(&block.restriction.vertices),
            Self::U64(block) => Slots::of(&block.restriction.vertices),
        }
    }

    /// The block, if its parts agree on its `n` vertices.
    pub(super) fn checked(self, n: usize) -> Option<Storage<T>> {
        match self {
            Self::U32(block) => block.checked(n).map(Storage::U32),
            Self::U64(block) => block.checked(n).map(Storage::U64),
        }
    }
}

#[derive(Deserialize)]
#[serde(
    rename = "Block",
    bound(deserialize = "T: Real + Deserialize<'de>, I: Index + Deserialize<'de>")
)]
pub(super) struct BlockWire<T, I> {
    restriction: RestrictionWire<T, I>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
    grounding_unproven: bool,
}

impl<T: Real, I: Index> BlockWire<T, I> {
    fn checked(self, n: usize) -> Option<Block<T, I>> {
        let block = Block {
            congruence: Box::new(self.restriction.congruence),
            elimination: self.elimination,
            cholesky: self.cholesky,
            grounding_unproven: self.grounding_unproven,
        };
        block.checked(n)
    }
}

#[derive(Deserialize)]
#[serde(
    rename = "Restriction",
    bound(deserialize = "T: Real + Deserialize<'de>, I: Index + Deserialize<'de>")
)]
struct RestrictionWire<T, I> {
    vertices: Selection<I>,
    congruence: Congruence<T>,
}

#[derive(Deserialize)]
#[serde(
    rename = "Isolated",
    bound(deserialize = "T: Real + Deserialize<'de>"),
    deny_unknown_fields
)]
pub(super) struct IsolatedWire<T> {
    vertex: u64,
    #[serde(deserialize_with = "super::inverse_of_a_pivot")]
    inverse_pivot: T,
}

impl<T> IsolatedWire<T> {
    /// `None` past `usize`, which no factored matrix indexes.
    pub(super) fn vertex(&self) -> Option<usize> {
        usize::try_from(self.vertex).ok()
    }

    pub(super) fn isolated(self) -> Isolated<T> {
        Isolated {
            inverse_pivot: self.inverse_pivot,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_test::{assert_ser_tokens, Token};

    use super::{Slots, Vertices};

    /// postcard writes a `u32` and a `u64` of one value alike, so only the serializer's calls show the width.
    #[test]
    fn vertices_are_written_in_the_blocks_index_type() {
        let run = Slots::Run { start: 3, len: 2 };
        let seq = |vertices: [Token; 2]| {
            [
                &[Token::Seq { len: Some(2) }],
                &vertices[..],
                &[Token::SeqEnd],
            ]
            .concat()
        };
        assert_ser_tokens(
            &Vertices::<u32>::new(run),
            &seq([Token::U32(3), Token::U32(4)]),
        );
        assert_ser_tokens(
            &Vertices::<u64>::new(run),
            &seq([Token::U64(3), Token::U64(4)]),
        );
        let listed = Slots::U64(&[u32::MAX as u64 + 1, 0]);
        assert_ser_tokens(
            &Vertices::<u64>::new(listed),
            &seq([Token::U64(u32::MAX as u64 + 1), Token::U64(0)]),
        );
    }
}
