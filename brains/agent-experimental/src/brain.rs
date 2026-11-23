use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use std::path::Path;

use crate::{
    AGENTS, BatchResult, FEATURES, UNITS, agent::Agent, calculator::Calculator, teacher::Teacher,
};

pub struct Brain {
    variables: VarMap,
    agents: Vec<Agent>,
}

impl Brain {
    pub fn new(device: &Device, classes: usize, lanes: usize) -> Result<Self> {
        if !(1..=2).contains(&lanes) {
            bail!("Expected 1 or 2 lanes");
        }

        let variables = VarMap::new();

        let builder = VarBuilder::from_varmap(&variables, DType::F32, device);

        let mut agents = Vec::with_capacity(AGENTS * lanes);

        for lane in 0..lanes {
            for index in 0..AGENTS {
                let input = builder.get_with_hints(
                    (FEATURES, UNITS),
                    &format!("lane_{lane}_agent_{index}_input"),
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.05,
                    },
                )?;

                let output = builder.get_with_hints(
                    (UNITS, classes),
                    &format!("lane_{lane}_agent_{index}_output"),
                    Init::Randn {
                        mean: 0.0,
                        stdev: 0.05,
                    },
                )?;

                let memory = builder.get_with_hints(
                    (UNITS,),
                    &format!("lane_{lane}_agent_{index}_memory"),
                    Init::Const(1.0),
                )?;

                agents.push(Agent::new(input, output, memory, lane, index));
            }
        }

        Ok(Self { variables, agents })
    }

    pub fn device(&self) -> &Device {
        self.agents[0].device()
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let mut total = self.agents[0].contribution(inputs)?;

        for agent in &self.agents[1..] {
            let contribution = agent.contribution(inputs)?;
            total = (&total + &contribution)?;
        }

        // Fusion interne des contributions des voies.
        Ok(total.affine(1.0 / self.agents.len() as f64, 0.0)?)
    }

    fn step(&self, inputs: &Tensor, targets: &Tensor, rate: f64) -> Result<(f32, f32)> {
        let logits = self.forward(inputs)?;
        let before = Calculator::error(&logits, targets)?;

        // On conserve les paramètres précédents en mémoire
        // afin de mesurer l'effet propre de chaque correction.
        let original: Vec<(Tensor, Tensor)> = self
            .agents
            .iter()
            .map(|agent| {
                Ok((
                    agent.input().affine(1.0, 0.0)?,
                    agent.output().affine(1.0, 0.0)?,
                ))
            })
            .collect::<Result<_>>()?;

        Teacher::teach(
            &self.variables,
            &self.agents,
            inputs,
            targets,
            &logits,
            rate,
        )?;

        let after_logits = self.forward(inputs)?;
        let after = Calculator::error(&after_logits, targets)?;
        let expected = Calculator::expected(&after_logits, targets)?;

        for (agent, (input, output)) in self.agents.iter().zip(original.iter()) {
            agent.remember(
                &self.variables,
                inputs,
                input,
                output,
                &after_logits,
                &expected,
                after,
                self.agents.len(),
            )?;
        }

        Ok((before, after))
    }

    pub fn train_batch(&self, inputs: &Tensor, targets: &Tensor, rate: f64) -> Result<BatchResult> {
        let (initial, after_first) = self.step(inputs, targets, rate)?;

        let (_, after_second) = self.step(inputs, targets, rate)?;

        Ok(BatchResult::new(initial, after_first, after_second))
    }

    pub fn predictions(&self, inputs: &Tensor) -> Result<Vec<u32>> {
        Ok(self.forward(inputs)?.argmax(1)?.to_vec1::<u32>()?)
    }

    pub fn probabilities(&self, inputs: &Tensor) -> Result<Vec<Vec<f32>>> {
        Ok(candle_nn::ops::softmax(&self.forward(inputs)?, 1)?.to_vec2::<f32>()?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let temporary = path.with_extension("safetensors.tmp");

        if temporary.exists() {
            bail!("Temporary checkpoint already exists");
        }

        let result = (|| -> Result<()> {
            self.variables.save(&temporary)?;
            std::fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }

        result
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        Ok(self.variables.load(path)?)
    }
}
