use super::{Example, BATCH};
use anyhow::Result;
use associative_circuits::{forward, FEATURES};
use candle_core::{Device, Tensor};

pub(super) fn batch(data: &[Example], device: &Device) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(data.len() * FEATURES);
    let mut targets = Vec::with_capacity(data.len());

    for example in data {
        inputs.extend_from_slice(&example.features);
        targets.push(example.label);
    }

    Ok((
        Tensor::from_vec(inputs, (data.len(), FEATURES), device)?,
        Tensor::from_vec(targets, data.len(), device)?,
    ))
}

pub(super) fn accuracy(
    data: &[Example],
    keys: &Tensor,
    values: &Tensor,
    device: &Device,
) -> Result<f32> {
    let mut correct = 0usize;

    for chunk in data.chunks(BATCH) {
        let (inputs, _) = batch(chunk, device)?;
        let predicted = forward(&inputs, keys, values)?
            .argmax(1)?
            .to_vec1::<u32>()?;

        correct += predicted
            .iter()
            .zip(chunk.iter())
            .filter(|(prediction, example)| **prediction == example.label)
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}
