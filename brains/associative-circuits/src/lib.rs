use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};

pub const FEATURES: usize = 1024;

pub fn encode(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let mut vector = vec![0f32; FEATURES];

    for gram in lower.as_bytes().windows(3) {
        let mut hash = 2166136261u32;

        for byte in gram {
            hash = (hash ^ *byte as u32).wrapping_mul(16777619);
        }

        vector[(hash as usize) % FEATURES] += 1.0;
    }

    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }

    vector
}

pub fn forward(inputs: &Tensor, keys: &Tensor, values: &Tensor) -> candle_core::Result<Tensor> {
    inputs.matmul(keys)?.relu()?.matmul(values)
}

pub fn setup(device: &Device, classes: usize, circuits: usize) -> Result<(VarMap, Tensor, Tensor)> {
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

    Ok((vars, keys, values))
}
