//! The path Laplacian 0-1-2-3, held once for the suites that rewrap it in their own types.

pub const N: u32 = 4;
pub const ROW_PTRS: [usize; 5] = [0, 2, 5, 8, 10];
pub const COL_INDICES: [usize; 10] = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
pub const VALUES: [f64; 10] = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
