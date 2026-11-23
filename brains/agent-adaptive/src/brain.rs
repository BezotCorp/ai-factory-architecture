use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use std::path::Path;

use crate::{
    AGENTS, FEATURES, MAX_MODEL_BYTES, UNITS, agent::Agent, calculator::Calculator,
    teacher::Teacher,
};

pub struct Brain {
    variables: VarMap,
    agents: Vec<Agent>,
}

impl Brain {
    pub fn new(device: &Device, classes: usize) -> Result<Self> {
        let bytes = (AGENTS * (FEATURES * UNITS + UNITS * classes + UNITS) * 4) as u64;

        if bytes > MAX_MODEL_BYTES {
            bail!("Parameter budget exceeded: {bytes}");
        }

        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, device);

        let mut agents = Vec::with_capacity(AGENTS);

        for index in 0..AGENTS {
            let input = builder.get_with_hints(
                (FEATURES, UNITS),
                &format!("agent_{index}_input"),
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
            )?;

            let output = builder.get_with_hints(
                (UNITS, classes),
                &format!("agent_{index}_output"),
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
            )?;

            let memory = builder.get_with_hints(
                (UNITS,),
                &format!("agent_{index}_memory"),
                Init::Const(1.0),
            )?;

            agents.push(Agent::new(input, output, memory, index));
        }

        Ok(Self { variables, agents })
    }

    pub fn device(&self) -> &Device {
        self.agents[0].device()
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let mut outputs = self.agents.iter().map(|agent| agent.forward(inputs));

        let mut sum = outputs
            .next()
            .ok_or_else(|| anyhow::anyhow!("No adaptive agents"))??;

        for output in outputs {
            sum = (&sum + &output?)?;
        }

        Ok(sum.affine(1.0 / AGENTS as f64, 0.0)?)
    }

    pub fn train_batch(&self, inputs: &Tensor, targets: &Tensor, rate: f64) -> Result<f32> {
        let before_logits = self.forward(inputs)?;
        let before = Calculator::error(&before_logits, targets)?;

        let importances = Teacher::teach(
            &self.variables,
            &self.agents,
            inputs,
            targets,
            &before_logits,
            rate,
        )?;

        let after_logits = self.forward(inputs)?;
        let after = Calculator::error(&after_logits, targets)?;
        let reward = Calculator::reward(before, after);

        Teacher::remember(&self.variables, &self.agents, &importances, reward)?;

        Ok(after)
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

            let size = std::fs::metadata(&temporary)?.len();
            if size > MAX_MODEL_BYTES {
                bail!("Checkpoint exceeds size budget: {size}");
            }

            std::fs::rename(&temporary, path)?;
            Ok(())
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }

        result
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        if std::fs::metadata(path)?.len() > MAX_MODEL_BYTES {
            bail!("Checkpoint exceeds size budget");
        }

        Ok(self.variables.load(path)?)
    }
}
