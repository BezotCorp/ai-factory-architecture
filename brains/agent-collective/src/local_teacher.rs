use crate::{
    error_calculator::ErrorCalculator,
    expert::Expert,
};
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;

pub(super) struct LocalTeacher;

impl LocalTeacher {
    pub(super) fn teach(
        variables: &VarMap,
        experts: &[Expert],
        inputs: &Tensor,
        targets: &Tensor,
        logits: &Tensor,
        learning_rate: f64,
    ) -> Result<f32> {
        // Le calculateur fournit l'erreur au professeur.
        let signal = ErrorCalculator::signal(logits, targets)?;

        let loss = ErrorCalculator::measure(logits, targets)?
            .to_scalar::<f32>()?;

        // Aucun agent n'est exclu de l'enseignement.
        for (index, expert) in experts.iter().enumerate() {
            expert.learn(
                variables,
                index,
                inputs,
                &signal,
                learning_rate,
            )?;
        }

        Ok(loss)
    }
}
