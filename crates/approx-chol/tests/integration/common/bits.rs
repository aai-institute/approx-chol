pub fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

/// A float's bits with every NaN as the one canonical NaN: an apply's contract leaves a NaN's sign and payload aside (#437).
pub trait Canonical: Copy {
    type Bits: PartialEq + std::fmt::Debug;

    fn canonical(self) -> Self::Bits;
}

impl Canonical for f64 {
    type Bits = u64;

    fn canonical(self) -> u64 {
        match self.is_nan() {
            true => f64::NAN.to_bits(),
            false => self.to_bits(),
        }
    }
}

impl Canonical for f32 {
    type Bits = u32;

    fn canonical(self) -> u32 {
        match self.is_nan() {
            true => f32::NAN.to_bits(),
            false => self.to_bits(),
        }
    }
}

/// Each entry's [`Canonical`] bits.
pub fn canonical_bits<T: Canonical>(x: &[T]) -> Vec<T::Bits> {
    x.iter().map(|&x| x.canonical()).collect()
}
