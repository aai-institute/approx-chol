// Each bench compiles this separately, so a helper only some use would read as dead.
#![allow(dead_code, unused_imports)]

#[path = "../../tests/common/grid.rs"]
pub mod grid;

pub use grid::grid_laplacian;
