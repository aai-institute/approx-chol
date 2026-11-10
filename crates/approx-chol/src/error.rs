/// Why a valid input did not reduce: a block too large, or a sum computed from it outside the representable range; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The block passes `1/ε` vertices: floor weights in series would fall below `MIN_POSITIVE`.
    TooLarge {
        /// The component's lowest vertex.
        vertex: usize,
    },
    /// A scale, a scaled or assembled entry, a diagonal, a row sum or a ground computed at row `vertex` left the representable range.
    OutOfRange {
        /// The offending row.
        vertex: usize,
    },
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooLarge { vertex } => {
                write!(f, "the block of vertex {vertex} passes 1/ε vertices")
            }
            Self::OutOfRange { vertex } => {
                write!(
                    f,
                    "an entry at vertex {vertex} is outside the representable range"
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}
