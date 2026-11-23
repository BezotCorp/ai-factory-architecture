use std::{collections::BTreeMap, path::Path};

use crate::adaptive::adaptive_encoding::encode;
use crate::instance::Instance;
use agent_adaptive::{Brain, FEATURES};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use massive_loader::Row;

use crate::adaptive::{
    config::{BATCH, EPOCHS},
    example::Example,
};

fn examples(rows: &[Row], labels: &BTreeMap<String, u32>) -> Result<Vec<Example>> {
    rows.iter()
        .map(|row| {
            Ok(Example {
                features: encode(&row.text),
                target: *labels.get(&row.intent).context("Unknown intent")?,
            })
        })
        .collect()
}

fn batch(data: &[Example], device: &Device) -> Result<(Tensor, Tensor)> {
    let mut inputs = Vec::with_capacity(data.len() * FEATURES);
    let mut targets = Vec::with_capacity(data.len());

    for example in data {
        inputs.extend_from_slice(&example.features);
        targets.push(example.target);
    }

    Ok((
        Tensor::from_vec(inputs, (data.len(), FEATURES), device)?,
        Tensor::from_vec(targets, data.len(), device)?,
    ))
}

fn accuracy(data: &[Example], brain: &Brain, device: &Device) -> Result<f32> {
    let mut correct = 0usize;

    for chunk in data.chunks(BATCH) {
        let (inputs, _) = batch(chunk, device)?;

        correct += brain
            .predictions(&inputs)?
            .iter()
            .zip(chunk)
            .filter(|(prediction, example)| **prediction == example.target)
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}

pub(in crate::adaptive) fn train(root: &Path, device: &Device) -> Result<()> {
    // Instance-specific storage: legacy checkpoints remain untouched.

    let dataset = massive_loader::load()?;
    let mut labels = BTreeMap::new();

    for row in dataset
        .train
        .iter()
        .chain(&dataset.validation)
        .chain(&dataset.test)
    {
        labels.entry(row.intent.clone()).or_insert(0u32);
    }

    for (index, id) in labels.values_mut().enumerate() {
        *id = index as u32;
    }

    let train = examples(&dataset.train, &labels)?;
    let validation = examples(&dataset.validation, &labels)?;
    let test = examples(&dataset.test, &labels)?;
    drop(dataset);

    let mut instance = Instance::create(
        root,
        "agent-adaptive",
        None,
        serde_json::json!({
            "classes": labels.len(),
            "features": FEATURES,
            "epochs": EPOCHS,
            "batch": BATCH,
            "learning_rate": 0.3
        }),
    )?;

    let label_file = instance.labels();
    let mut best_weights = None;

    let mut brain = Brain::new(device, labels.len())?;
    let mut best = -1.0f32;
    let mut best_epoch = 0;

    for epoch in 1..=EPOCHS {
        let mut loss = 0.0;
        let mut steps = 0;

        for chunk in train.chunks(BATCH) {
            let (inputs, targets) = batch(chunk, device)?;
            loss += brain.train_batch(&inputs, &targets, 0.3)?;
            steps += 1;
        }

        let validation_score = accuracy(&validation, &brain, device)?;

        println!(
            "Epoch {epoch:02}/{EPOCHS} | loss {:.4} | validation {:.2}%",
            loss / steps as f32,
            validation_score,
        );

        if validation_score > best {
            best = validation_score;
            best_epoch = epoch;
            let checkpoint = instance.checkpoint(epoch);
            brain.save(&checkpoint)?;
            instance.record(epoch, &checkpoint, validation_score)?;
            best_weights = Some(checkpoint);
        }
    }

    let weights = best_weights.ok_or_else(|| anyhow::anyhow!("No best checkpoint"))?;
    brain.load(&weights)?;
    let test_score = accuracy(&test, &brain, device)?;

    println!(
        "Best epoch: {best_epoch} | validation {best:.2}% | test {:.2}%",
        test_score,
    );

    let names: Vec<&str> = labels.keys().map(String::as_str).collect();

    std::fs::write(&label_file, serde_json::to_vec_pretty(&names)?)?;

    println!("Weights: {} bytes", std::fs::metadata(weights)?.len());
    instance.finish(best_epoch, best, test_score)?;

    Ok(())
}
