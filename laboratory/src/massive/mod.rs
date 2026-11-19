mod classifier;
mod compare;
mod example;
mod predict;
mod train;
mod training_data;

use example::Example;
const CIRCUITS: usize = 128;
const BATCH: usize = 128;
const EPOCHS: usize = 20;

pub use compare::compare;
pub use predict::predict;
pub use train::train;
