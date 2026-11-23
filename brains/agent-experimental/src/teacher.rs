use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;

use crate::{agent::Agent, calculator::Calculator};

pub(crate) struct Teacher;

impl Teacher {
    pub(crate) fn teach(
        variables: &VarMap,
        agents: &[Agent],
        inputs: &Tensor,
        targets: &Tensor,
        logits: &Tensor,
        rate: f64,
    ) -> Result<()> {
        let expected = Calculator::expected(logits, targets)?;

        // Les diagnostics utilisent la même sortie de référence.
        // Chaque agent applique ses corrections individuelles.
        for agent in agents {
            agent.teach(variables, inputs, logits, &expected, rate, agents.len())?;
        }

        Ok(())
    }
}
