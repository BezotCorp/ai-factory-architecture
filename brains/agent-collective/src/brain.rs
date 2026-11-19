use crate::{
    error_calculator::ErrorCalculator,
    expert::Expert,
    teacher::Teacher,
    Strategy, AGENTS, FEATURES, UNITS, MAX_MODEL_BYTES,
};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    AdamW, Init, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
use std::path::Path;
use crate::local_teacher::LocalTeacher;

pub struct Brain {
    vars: VarMap,
    experts: Vec<Expert>,
}

impl Brain {
    pub fn new(device: &Device, classes: usize) -> Result<Self> {
        let bytes = (AGENTS * (
            FEATURES * UNITS + UNITS * classes
        ) * 4) as u64;

        if bytes > MAX_MODEL_BYTES {
            bail!("Parameter budget exceeded: {bytes} bytes");
        }

        let vars = VarMap::new();
        let builder = VarBuilder::from_varmap(
            &vars, DType::F32, device,
        );

        let mut experts = Vec::with_capacity(AGENTS);

        for i in 0..AGENTS {
            let input = builder.get_with_hints(
                (FEATURES, UNITS),
                &format!("agent_{i}_input"),
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
            )?;

            let output = builder.get_with_hints(
                (UNITS, classes),
                &format!("agent_{i}_output"),
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.05,
                },
            )?;

            experts.push(Expert { input, output });
        }

        Ok(Self { vars, experts })
    }

    pub fn device(&self) -> &Device {
        self.experts[0].input.device()
    }

    fn outputs(&self, inputs: &Tensor) -> Result<Vec<Tensor>> {
        Ok(self.experts.iter()
            .map(|expert| expert.forward(inputs))
            .collect::<candle_core::Result<Vec<_>>>()?)
    }

    pub fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let mut outputs = self.outputs(inputs)?.into_iter();
        let mut result = outputs.next().expect("No experts");

        for output in outputs {
            result = (&result + &output)?;
        }

        Ok(result.affine(1.0 / AGENTS as f64, 0.0)?)
    }

    pub fn optimizer(&self, lr: f64) -> Result<AdamW> {
        Ok(AdamW::new(
            self.vars.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?)
    }

    pub fn train_batch(
        &self,
        optimizer: &mut AdamW,
        inputs: &Tensor,
        targets: &Tensor,
        strategy: Strategy,
    ) -> Result<f32> {
        let losses = self.outputs(inputs)?
            .iter()
            .map(|logits| {
                ErrorCalculator::measure(logits, targets)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let lesson = Teacher::lesson(&losses, strategy)?;
        let value = lesson.to_scalar::<f32>()?;
        optimizer.backward_step(&lesson)?;
        Ok(value)
    }


    pub fn learn_locally(
        &self,
        inputs: &Tensor,
        targets: &Tensor,
        learning_rate: f64,
    ) -> Result<f32> {
        let logits = self.forward(inputs)?;

        LocalTeacher::teach(
            &self.vars,
            &self.experts,
            inputs,
            targets,
            &logits,
            learning_rate,
        )
    }

    pub fn predictions(&self, inputs: &Tensor) -> Result<Vec<u32>> {
        Ok(self.forward(inputs)?
            .argmax(1)?
            .to_vec1::<u32>()?)
    }

    pub fn probabilities(
        &self,
        inputs: &Tensor,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(candle_nn::ops::softmax(
            &self.forward(inputs)?, 1,
        )?.to_vec2::<f32>()?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.exists() {
            bail!("Refusing to overwrite existing model");
        }

        let temporary = path.with_extension("safetensors.tmp");
        if temporary.exists() {
            bail!("Temporary model already exists");
        }

        let result = (|| -> Result<()> {
            self.vars.save(&temporary)?;

            let size = std::fs::metadata(&temporary)?.len();
            if size > MAX_MODEL_BYTES {
                bail!("Model exceeds budget: {size} bytes");
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
            bail!("Model exceeds budget");
        }

        Ok(self.vars.load(path)?)
    }
}
