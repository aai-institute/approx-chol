use crate::entry::{by_type, Factored};

impl Factored {
    pub fn n(&self) -> usize {
        by_type!(&self.factor, factor => factor.n())
    }
}
