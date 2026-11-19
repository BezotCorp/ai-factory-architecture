use super::{
    training_data::{accuracy, batch, encode_rows},
    BATCH, EPOCHS,
};
use anyhow::Result;
use associative_circuits::Brain;
use candle_core::Device;
use std::{collections::BTreeMap, path::Path};

pub fn train(dir: &Path, device: &Device) -> Result<()> {
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

    println!(
        "Train {} | Validation {} | Test {} | Classes {}",
        train.len(),
        validation.len(),
        test.len(),
        labels.len()
    );

    let mut brain = Brain::new(device, labels.len(), super::CIRCUITS)?;
    let mut optimizer = brain.optimizer(0.003)?;

    let mut best = -1f32;
    let mut best_epoch = 0;
    let weights = dir.join("artifacts/associative-circuits/massive.safetensors");

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0f32;
        let mut steps = 0usize;

        for chunk in train.chunks(BATCH) {
            let (inputs, targets) = batch(chunk, device)?;

            total_loss += brain.train_batch(&mut optimizer, &inputs, &targets)?;
            steps += 1;
        }

        let score = accuracy(&validation, &brain, device)?;

        println!(
            "Epoch {epoch:02}/{EPOCHS} | loss {:.4} | validation {:.2}%",
            total_loss / steps as f32,
            score
        );

        if score > best {
            best = score;
            best_epoch = epoch;
            brain.save(&weights)?;
        }
    }

    brain.load(&weights)?;

    println!("Best epoch: {best_epoch} | validation {best:.2}%");

    println!(
        "FINAL TEST: {:.2}%",
        accuracy(&test, &brain, device)?
    );

    let names: Vec<&str> = labels.keys().map(String::as_str).collect();

    std::fs::write(
        dir.join("artifacts/associative-circuits/massive-labels.json"),
        serde_json::to_vec_pretty(&names)?,
    )?;

    println!(
        "Model weights: {} bytes",
        std::fs::metadata(&weights)?.len()
    );

    Ok(())
}
