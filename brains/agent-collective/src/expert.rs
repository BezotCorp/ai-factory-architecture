use candle_core::{DType, Tensor};
use candle_nn::VarMap;
use anyhow::{Context, Result};

pub(super) struct Expert {
    pub(super) input: Tensor,
    pub(super) output: Tensor,
}

impl Expert {

    pub(super) fn learn(
        &self,
        variables: &VarMap,
        index: usize,
        inputs: &Tensor,
        signal: &Tensor,
        rate: f64,
    ) -> Result<()> {
        let (input_var, output_var) = {
            let map = variables.data().lock().unwrap();

            (
                map.get(&format!("agent_{index}_input"))
                    .cloned()
                    .context("Missing agent input parameters")?,
                map.get(&format!("agent_{index}_output"))
                    .cloned()
                    .context("Missing agent output parameters")?,
            )
        };

        let batch = inputs.dim(0)? as f64;
        let classes = signal.dim(1)?;
        let units = self.input.dim(1)?;

        // Activité locale des unités de cet agent.
        let activity = inputs.matmul(&self.input)?.relu()?;

        // Rétroaction déterministe, différente selon l'agent.
        // Elle n'est ni entraînée ni sauvegardée.
        let mut feedback = Vec::with_capacity(classes * units);
        let mut state = 0x9e3779b9u32
            .wrapping_mul((index as u32).wrapping_add(1));

        for _ in 0..classes * units {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;

            let value = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            feedback.push(value / (classes as f32).sqrt());
        }

        let feedback = Tensor::from_vec(
            feedback,
            (classes, units),
            inputs.device(),
        )?;

        // Chaque unité utilise son activité et le signal reçu.
        let gate = activity.gt(0.0)?.to_dtype(DType::F32)?;

        let correction = signal
            .matmul(&feedback)?
            .mul(&gate)?;

        // Ajustement des connexions vers les sorties.
        let output_delta = activity
            .t()?
            .matmul(signal)?
            .affine(rate / batch, 0.0)?
            .clamp(-0.01, 0.01)?;

        // Ajustement des connexions d'entrée.
        let input_delta = inputs
            .t()?
            .matmul(&correction)?
            .affine(rate / batch, 0.0)?
            .clamp(-0.01, 0.01)?;

        let next_input =
            (&input_var.as_tensor().detach() + &input_delta)?;

        let next_output =
            (&output_var.as_tensor().detach() + &output_delta)?;

        input_var.set(&next_input)?;
        output_var.set(&next_output)?;

        Ok(())
    }

    pub(super) fn forward(
        &self,
        x: &Tensor,
    ) -> candle_core::Result<Tensor> {
        x.matmul(&self.input)?
            .relu()?
            .matmul(&self.output)
    }
}
