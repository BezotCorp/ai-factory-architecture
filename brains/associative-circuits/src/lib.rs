mod brain;
mod encode;

pub use brain::Brain;
pub use encode::encode;

pub const FEATURES: usize = 1024;
