#[path = "../../tests/common/grid.rs"]
pub mod grid;
#[path = "../../tests/common/panic_ok.rs"]
mod panic_ok;

pub use grid::grid_laplacian;
pub use panic_ok::OrPanic;
