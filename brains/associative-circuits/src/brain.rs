use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::FEATURES;

pub struct Brain {
    vars: VarMap,
    keys: Tensor,
    values: Tensor,
}

impl Brain {
    pub fn new(device: &Device, classes: usize, circuits: usize) -> Result<Self> {
        let vars = VarMap::new();
        let builder = VarBuilder::from_varmap(&vars, DType::F32, device);

        let keys = builder.get_with_hints(
            (FEATURES, circuits),
            "associations",
            Init::Randn {
                mean: 0.0,
                stdev: 0.05,
            },
        )?;

        let values = builder.get_with_hints(
            (circuits, classes),
            "responses",
            Init::Randn {
                mean: 0.0,
                stdev: 0.05,
            },
        )?;

        Ok(Self { vars, keys, values })
    }

    pub fn device(&self) -> &Device {
        self.keys.device()
    }

    pub fn forward(&self, inputs: &Tensor) -> candle_core::Result<Tensor> {
        inputs.matmul(&self.keys)?.relu()?.matmul(&self.values)
    }

    pub fn optimizer(&self, learning_rate: f64) -> Result<AdamW> {
        Ok(AdamW::new(
            self.vars.all_vars(),
            ParamsAdamW {
                lr: learning_rate,
                ..Default::default()
            },
        )?)
    }

    pub fn train_batch(
        &self,
        optimizer: &mut AdamW,
        inputs: &Tensor,
        targets: &Tensor,
    ) -> Result<f32> {
        let logits = self.forward(inputs)?;
        let error = loss::cross_entropy(&logits, targets)?;
        let value = error.to_scalar::<f32>()?;
        optimizer.backward_step(&error)?;
        Ok(value)
    }

    pub fn predict_batch(&self, inputs: &Tensor) -> Result<Vec<u32>> {
        Ok(self.forward(inputs)?.argmax(1)?.to_vec1::<u32>()?)
    }

    pub fn probabilities(&self, inputs: &Tensor) -> Result<Vec<Vec<f32>>> {
        Ok(
            candle_nn::ops::softmax(&self.forward(inputs)?, 1)?
                .to_vec2::<f32>()?
        )
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(self.vars.save(path)?)
    }

    pub fn load(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Ok(self.vars.load(path)?)
    }
}
