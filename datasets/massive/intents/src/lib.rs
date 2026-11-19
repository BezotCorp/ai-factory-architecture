mod case;
mod evaluation;
mod prediction;
mod predictor;
mod suite;

pub use case::Case;
pub use evaluation::{evaluate, load_suite};
pub use prediction::Prediction;
pub use predictor::Predictor;
pub use suite::Suite;

mod names;
pub use names::display;
