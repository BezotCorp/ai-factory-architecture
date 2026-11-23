use crate::massive::{config::BATCH, example::Example};
use anyhow::{Context, Result};
use associative_circuits::{Brain, FEATURES, encode};
use candle_core::{Device, Tensor};
use massive_loader::Row;
use std::collections::BTreeMap;

pub(in crate::massive) fn encode_rows(
    rows: &[Row],
    labels: &BTreeMap<String, u32>,
) -> Result<Vec<Example>> {
    rows.iter()
        .map(|row| {
            Ok(Example {
                features: encode(&row.text),
                target: *labels.get(&row.intent).context("Unknown intent")?,
            })
        })
        .collect()
}

pub(in crate::massive) fn batch(rows: &[Example], device: &Device) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(rows.len() * FEATURES);
    let mut targets = Vec::with_capacity(rows.len());

    for row in rows {
        inputs.extend_from_slice(&row.features);
        targets.push(row.target);
    }

    Ok((
        Tensor::from_vec(inputs, (rows.len(), FEATURES), device)?,
        Tensor::from_vec(targets, rows.len(), device)?,
    ))
}

pub(in crate::massive) fn accuracy(
    data: &[Example],
    brain: &Brain,
    device: &Device,
) -> Result<f32> {
    let mut correct = 0usize;

    for chunk in data.chunks(BATCH) {
        let (inputs, _) = batch(chunk, device)?;

        let predictions = brain.predict_batch(&inputs)?;

        correct += predictions
            .iter()
            .zip(chunk.iter())
            .filter(|(prediction, row)| **prediction == row.target)
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}
