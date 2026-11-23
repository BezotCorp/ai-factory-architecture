use crate::collective::{
    config::{BATCH, EPOCHS},
    example::Example,
};
use crate::instance::Instance;
use agent_collective::{Brain, FEATURES, Strategy};
use anyhow::{Context, Result};
use associative_circuits::encode;
use candle_core::{Device, Tensor};
use massive_loader::Row;
use std::{collections::BTreeMap, path::Path};

fn encode_rows(rows: &[Row], labels: &BTreeMap<String, u32>) -> Result<Vec<Example>> {
    rows.iter()
        .map(|row| {
            Ok(Example {
                features: encode(&row.text),
                target: *labels.get(&row.intent).context("Unknown intent")?,
            })
        })
        .collect()
}

fn batch(rows: &[Example], device: &Device) -> Result<(Tensor, Tensor)> {
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

fn accuracy(data: &[Example], brain: &Brain, device: &Device) -> Result<f32> {
    let mut correct = 0usize;

    for chunk in data.chunks(BATCH) {
        let (inputs, _) = batch(chunk, device)?;
        let predictions = brain.predictions(&inputs)?;

        correct += predictions
            .iter()
            .zip(chunk.iter())
            .filter(|(prediction, row)| **prediction == row.target)
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}

pub(in crate::collective) fn train(root: &Path, device: &Device, strategy: Strategy) -> Result<()> {
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

    let train = encode_rows(&dataset.train, &labels)?;
    let validation = encode_rows(&dataset.validation, &labels)?;
    let test = encode_rows(&dataset.test, &labels)?;
    drop(dataset);

    let mut instance = Instance::create(
        root,
        "agent-collective",
        Some(strategy.name()),
        serde_json::json!({
            "classes": labels.len(),
            "features": FEATURES,
            "agents": agent_collective::AGENTS,
            "units": agent_collective::UNITS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "learning_rate": 0.003
        }),
    )?;

    let label_file = instance.labels();
    let mut best_weights = None;

    let brain = Brain::new(device, labels.len())?;
    let mut optimizer = brain.optimizer(0.003)?;

    let mut best = -1.0f32;
    let mut best_epoch = 0usize;

    println!("Training collective {}", strategy.name());

    for epoch in 1..=EPOCHS {
        let mut total = 0.0f32;
        let mut steps = 0usize;

        for chunk in train.chunks(BATCH) {
            let (inputs, targets) = batch(chunk, device)?;
            total += if matches!(strategy, Strategy::Local) {
                brain.learn_locally(&inputs, &targets, 0.003)?
            } else {
                brain.train_batch(&mut optimizer, &inputs, &targets, strategy)?
            };
            steps += 1;
        }

        let score = accuracy(&validation, &brain, device)?;
        println!(
            "Epoch {epoch:02}/{EPOCHS} | loss {:.4} | validation {:.2}%",
            total / steps as f32,
            score,
        );

        if score > best {
            best = score;
            best_epoch = epoch;

            let checkpoint = instance.checkpoint(epoch);
            brain.save(&checkpoint)?;
            instance.record(epoch, &checkpoint, score)?;
            best_weights = Some(checkpoint);
        }
    }

    let mut brain = brain;
    let weights = best_weights.ok_or_else(|| anyhow::anyhow!("No best checkpoint"))?;
    brain.load(&weights)?;
    let test_score = accuracy(&test, &brain, device)?;

    println!(
        "Best epoch {best_epoch} | validation {best:.2}% | test {:.2}%",
        test_score
    );

    let names: Vec<&str> = labels.keys().map(String::as_str).collect();

    std::fs::write(&label_file, serde_json::to_vec_pretty(&names)?)?;

    println!("Weights: {} bytes", std::fs::metadata(&weights)?.len());
    instance.finish(best_epoch, best, test_score)?;

    Ok(())
}
