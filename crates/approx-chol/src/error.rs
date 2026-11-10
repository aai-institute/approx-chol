/// Why a valid input did not reduce: a block too large, or a sum computed from it outside the representable range; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The block passes `1/ε` vertices: floor weights in series would fall below `MIN_POSITIVE`.
    TooLarge {
        /// The component's lowest vertex.
        vertex: usize,
    },
    /// A value computed at row `vertex` left the representable range: an assembled entry, diagonal or row sum, a Gram's derived surplus, a relaxation's ratio or scale, a scaled entry at its stored row, or a ground; a block left with no representable ground names the first row its float-or-ground rule marks a ground, else its row of the largest residue.
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
                    "a value at vertex {vertex} is outside the representable range"
                )
            }
        }
    }
}

impl std::error::Error for BuildError {}
