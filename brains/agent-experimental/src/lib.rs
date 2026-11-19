mod agent;
mod brain;
mod calculator;
mod teacher;

pub use brain::{BatchResult, Brain};

pub const FEATURES: usize = 1024;
pub const AGENTS: usize = 3;
pub const UNITS: usize = 32;
