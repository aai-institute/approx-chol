// Reuse the canonical grid Laplacian helper from the test support tree, the
// same way `benches/common` does, instead of carrying a second copy here.
#[path = "../../tests/common/grid.rs"]
pub mod grid;

pub use grid::grid_laplacian;
