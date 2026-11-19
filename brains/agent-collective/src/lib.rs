mod brain;
mod local_teacher;
mod error_calculator;
mod expert;
mod regulator;
mod strategy;
mod teacher;

pub use brain::Brain;
pub use strategy::Strategy;

pub const FEATURES: usize = 1024;
pub const AGENTS: usize = 3;
pub const UNITS: usize = 32;
pub const MAX_MODEL_BYTES: u64 = 1_048_576;
