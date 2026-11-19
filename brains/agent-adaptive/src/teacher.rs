use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;

use crate::{agent::Agent, calculator::Calculator};

pub(super) struct Teacher;

impl Teacher {
    pub(super) fn teach(
        variables: &VarMap,
        agents: &[Agent],
        inputs: &Tensor,
        targets: &Tensor,
        logits: &Tensor,
        rate: f64,
    ) -> Result<Vec<Vec<f32>>> {
        let signal = Calculator::signal(logits, targets)?;

        // Tous les agents reçoivent un enseignement.
        // Chaque agent calcule et applique ses propres mises à jour.
        agents.iter()
            .map(|agent| {
                agent.learn(variables, inputs, &signal, rate)
            })
            .collect()
    }

    pub(super) fn remember(
        variables: &VarMap,
        agents: &[Agent],
        importances: &[Vec<f32>],
        reward: f32,
    ) -> Result<()> {
        for (agent, importance) in agents.iter().zip(importances) {
            agent.remember(variables, importance, reward)?;
        }

        Ok(())
    }
}
