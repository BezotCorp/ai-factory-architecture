mod agent;
mod brain;
mod calculator;
mod teacher;

pub use brain::Brain;

pub const FEATURES: usize = 1024;
pub const AGENTS: usize = 3;
pub const UNITS: usize = 32;
pub const MAX_MODEL_BYTES: u64 = 1_048_576;
