use crate::Prediction;
use anyhow::Result;

pub trait Predictor {
    fn name(&self) -> &str;
    fn predict(&self, input: &str, limit: usize) -> Result<Vec<Prediction>>;
}
