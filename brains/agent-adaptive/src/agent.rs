use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;

use crate::UNITS;

pub(crate) struct Agent {
    input: Tensor,
    output: Tensor,
    memory: Tensor,
    index: usize,
}

impl Agent {
    pub(crate) fn new(input: Tensor, output: Tensor, memory: Tensor, index: usize) -> Self {
        Self {
            input,
            output,
            memory,
            index,
        }
    }

    pub(crate) fn input(&self) -> &Tensor {
        &self.input
    }

    pub(crate) fn output(&self) -> &Tensor {
        &self.output
    }

    pub(crate) fn memory(&self) -> &Tensor {
        &self.memory
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn device(&self) -> &Device {
        self.input.device()
    }

    pub(crate) fn activity(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.input)?.relu()?)
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.activity(x)?.matmul(&self.output)?)
    }

    pub(crate) fn learn(
        &self,
        variables: &VarMap,
        inputs: &Tensor,
        signal: &Tensor,
        rate: f64,
    ) -> Result<Vec<f32>> {
        let (input_var, output_var) = {
            let map = variables
                .data()
                .lock()
                .map_err(|_| anyhow::anyhow!("Parameter storage lock poisoned"))?;
            (
                map.get(&format!("agent_{}_input", self.index))
                    .cloned()
                    .context("Input parameters missing")?,
                map.get(&format!("agent_{}_output", self.index))
                    .cloned()
                    .context("Output parameters missing")?,
            )
        };

        let batch = inputs.dim(0)? as f64;

        let activity = self.activity(inputs)?;
        let gate = activity.gt(0.0)?.to_dtype(candle_core::DType::F32)?;

        // Le calculateur attribue une direction de correction
        // à chaque unité via ses connexions de sortie.
        let influence = signal.matmul(&self.output.t()?)?;
        let correction = influence.mul(&gate)?.broadcast_mul(&self.memory)?;

        // Chaque agent ajuste lui-même ses deux ensembles
        // de connexions ; aucun appel à backward().
        let output_delta = activity
            .t()?
            .matmul(signal)?
            .affine(rate / batch, 0.0)?
            .clamp(-0.02, 0.02)?;

        let input_delta = inputs
            .t()?
            .matmul(&correction)?
            .affine(rate / batch, 0.0)?
            .clamp(-0.02, 0.02)?;

        let next_input = (input_var.as_tensor() + &input_delta)?;
        let next_output = (output_var.as_tensor() + &output_delta)?;

        input_var.set(&next_input)?;
        output_var.set(&next_output)?;

        // Importance estimée de chaque neurone sur ce batch.
        Ok(correction.sqr()?.mean(0)?.to_vec1::<f32>()?)
    }

    pub(crate) fn remember(
        &self,
        variables: &VarMap,
        importance: &[f32],
        reward: f32,
    ) -> Result<()> {
        let memory_var = {
            let map = variables
                .data()
                .lock()
                .map_err(|_| anyhow::anyhow!("Parameter storage lock poisoned"))?;
            map.get(&format!("agent_{}_memory", self.index))
                .cloned()
                .context("Neuron memory missing")?
        };

        let old = memory_var.as_tensor().to_vec1::<f32>()?;
        let average = importance.iter().sum::<f32>() / importance.len() as f32;

        let updated: Vec<f32> = old
            .iter()
            .zip(importance)
            .map(|(value, importance)| {
                let relevance = if average > 1e-12 {
                    (importance / average).clamp(0.0, 3.0)
                } else {
                    0.0
                };

                (value * (1.0 + 0.05 * reward * relevance)).clamp(0.25, 4.0)
            })
            .collect();

        let tensor = Tensor::from_vec(updated, (UNITS,), self.device())?;

        memory_var.set(&tensor)?;
        Ok(())
    }
}
