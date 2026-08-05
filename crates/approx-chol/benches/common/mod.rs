// Each bench binary compiles this module separately, so a helper only some of
// them need would otherwise read as dead.
#![allow(dead_code, unused_imports)]

#[path = "../../tests/common/grid.rs"]
pub mod grid;

pub use grid::grid_laplacian;
