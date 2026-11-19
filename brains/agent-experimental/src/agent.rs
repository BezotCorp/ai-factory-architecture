use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;

use crate::{calculator::Calculator, UNITS};

pub(super) struct Agent {
    pub(super) input: Tensor,
    pub(super) output: Tensor,
    pub(super) memory: Tensor,
    pub(super) lane: usize,
    pub(super) index: usize,
}

impl Agent {
    fn variable(
        &self,
        variables: &VarMap,
        suffix: &str,
    ) -> Result<candle_core::Var> {
        variables
            .data()
            .lock()
            .unwrap()
            .get(&format!(
                "lane_{}_agent_{}_{}",
                self.lane,
                self.index,
                suffix
            ))
            .cloned()
            .with_context(|| format!("Missing {suffix} parameters"))
    }

    pub(super) fn device(&self) -> &Device {
        self.input.device()
    }

    pub(super) fn activity(
        &self,
        inputs: &Tensor,
    ) -> Result<Tensor> {
        Ok(inputs.matmul(&self.input)?.relu()?)
    }

    pub(super) fn contribution(
        &self,
        inputs: &Tensor,
    ) -> Result<Tensor> {
        Ok(self.activity(inputs)?.matmul(&self.output)?)
    }

    pub(super) fn teach(
        &self,
        variables: &VarMap,
        inputs: &Tensor,
        logits: &Tensor,
        expected: &Tensor,
        rate: f64,
        total_agents: usize,
    ) -> Result<()> {
        let activity = self.activity(inputs)?;
        let batch = inputs.dim(0)? as f64;
        let residual = Calculator::residual(logits, expected)?;
        let gains = self.memory.to_vec1::<f32>()?;

        let transposed_inputs = inputs.t()?;
        let mut inbound = Vec::with_capacity(UNITS);

        for (unit, gain) in gains.iter().enumerate() {
            let outgoing = self.output.narrow(0, unit, 1)?;

            let direction = Calculator::direction(
                logits,
                expected,
                &outgoing,
                total_agents,
            )?;

            let active = activity
                .narrow(1, unit, 1)?
                .gt(0.0)?
                .to_dtype(DType::F32)?;

            let local = direction
                .mul(&active)?
                .affine(*gain as f64, 0.0)?;

            let delta = transposed_inputs
                .matmul(&local)?
                .affine(rate / batch, 0.0)?
                .clamp(-0.02, 0.02)?;

            inbound.push(delta);
        }

        let input_delta = Tensor::cat(&inbound, 1)?;

        let gain_tensor = self.memory.reshape((UNITS, 1))?;

        let output_delta = activity
            .t()?
            .matmul(&residual)?
            .broadcast_mul(&gain_tensor)?
            .affine(
                rate / (batch * total_agents as f64),
                0.0,
            )?
            .clamp(-0.02, 0.02)?;

        let input_var = self.variable(variables, "input")?;
        let output_var = self.variable(variables, "output")?;

        input_var.set(
            &(input_var.as_tensor() + &input_delta)?
        )?;

        output_var.set(
            &(output_var.as_tensor() + &output_delta)?
        )?;

        Ok(())
    }

    // Pour chaque neurone, le calculateur mesure ce que
    // serait devenue l'erreur si sa correction n'avait
    // pas été appliquée, toutes les autres étant conservées.
    //
    // La mémoire individuelle évolue suivant cet effet.
    pub(super) fn remember(
        &self,
        variables: &VarMap,
        inputs: &Tensor,
        before_input: &Tensor,
        before_output: &Tensor,
        after_logits: &Tensor,
        expected: &Tensor,
        after_loss: f32,
        total_agents: usize,
    ) -> Result<()> {
        let old_activity = inputs.matmul(before_input)?.relu()?;
        let new_activity = self.activity(inputs)?;

        let mut gains = self.memory.to_vec1::<f32>()?;

        for (unit, gain) in gains.iter_mut().enumerate() {
            let old_contribution = old_activity
                .narrow(1, unit, 1)?
                .contiguous()?
                .matmul(&before_output.narrow(0, unit, 1)?)?;

            let new_contribution = new_activity
                .narrow(1, unit, 1)?
                .contiguous()?
                .matmul(&self.output.narrow(0, unit, 1)?)?;

            let delta = (&new_contribution - &old_contribution)?
                .affine(1.0 / total_agents as f64, 0.0)?;

            let without_correction = (
                after_logits - &delta
            )?;

            let counterfactual = Calculator::local_errors(
                &without_correction,
                expected,
            )?
            .mean_all()?
            .to_scalar::<f32>()?;

            let benefit = (
                counterfactual - after_loss
            ).clamp(-0.25, 0.25);

            *gain = (
                *gain * (0.5 * benefit).exp()
            ).clamp(0.1, 4.0);
        }

        let memory = self.variable(variables, "memory")?;

        memory.set(&Tensor::from_vec(
            gains,
            (UNITS,),
            self.device(),
        )?)?;

        Ok(())
    }
}
