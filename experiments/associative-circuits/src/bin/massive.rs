mod intent_names;

use anyhow::{Context, Result};
#[path = "massive_data.rs"]
mod data;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    loss, AdamW, Init, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::Path,
};

const FEATURES: usize = 1024;
const CIRCUITS: usize = 128;
const BATCH: usize = 128;
const EPOCHS: usize = 20;

#[derive(Deserialize)]
struct Row {
    text: String,
    intent: String,
}

#[derive(Deserialize)]
struct Dataset {
    train: Vec<Row>,
    validation: Vec<Row>,
    test: Vec<Row>,
}

struct Example {
    features: Vec<f32>,
    target: u32,
}

fn encode(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let mut vector = vec![0f32; FEATURES];

    for gram in lower.as_bytes().windows(3) {
        let mut hash = 2166136261u32;

        for byte in gram {
            hash = (hash ^ *byte as u32)
                .wrapping_mul(16777619);
        }

        vector[(hash as usize) % FEATURES] += 1.0;
    }

    let norm = vector
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();

    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }

    vector
}

fn encode_rows(
    rows: &[Row],
    labels: &BTreeMap<String, u32>,
) -> Result<Vec<Example>> {
    rows.iter()
        .map(|row| {
            Ok(Example {
                features: encode(&row.text),
                target: *labels
                    .get(&row.intent)
                    .context("Unknown intent")?,
            })
        })
        .collect()
}

fn batch(
    rows: &[Example],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut inputs =
        Vec::with_capacity(rows.len() * FEATURES);
    let mut targets = Vec::with_capacity(rows.len());

    for row in rows {
        inputs.extend_from_slice(&row.features);
        targets.push(row.target);
    }

    Ok((
        Tensor::from_vec(
            inputs,
            (rows.len(), FEATURES),
            device,
        )?,
        Tensor::from_vec(targets, rows.len(), device)?,
    ))
}

fn forward(
    inputs: &Tensor,
    keys: &Tensor,
    values: &Tensor,
) -> candle_core::Result<Tensor> {
    inputs.matmul(keys)?.relu()?.matmul(values)
}

fn accuracy(
    data: &[Example],
    keys: &Tensor,
    values: &Tensor,
    device: &Device,
) -> Result<f32> {
    let mut correct = 0usize;

    for chunk in data.chunks(BATCH) {
        let (inputs, _) = batch(chunk, device)?;

        let predictions = forward(
            &inputs,
            keys,
            values,
        )?
        .argmax(1)?
        .to_vec1::<u32>()?;

        correct += predictions
            .iter()
            .zip(chunk.iter())
            .filter(|(prediction, row)| {
                **prediction == row.target
            })
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}

fn setup(
    device: &Device,
    classes: usize,
) -> Result<(VarMap, Tensor, Tensor)> {
    let vars = VarMap::new();
    let builder =
        VarBuilder::from_varmap(&vars, DType::F32, device);

    let keys = builder.get_with_hints(
        (FEATURES, CIRCUITS),
        "associations",
        Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        },
    )?;

    let values = builder.get_with_hints(
        (CIRCUITS, classes),
        "responses",
        Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        },
    )?;

    Ok((vars, keys, values))
}

fn train(dir: &Path, device: &Device) -> Result<()> {
    let dataset = data::load()?;

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
    let validation =
        encode_rows(&dataset.validation, &labels)?;
    let test = encode_rows(&dataset.test, &labels)?;

    drop(dataset);

    println!(
        "Train {} | Validation {} | Test {} | Classes {}",
        train.len(),
        validation.len(),
        test.len(),
        labels.len()
    );

    let (mut vars, keys, values) =
        setup(device, labels.len())?;

    let mut optimizer = AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: 0.003,
            ..Default::default()
        },
    )?;

    let mut best = -1f32;
    let mut best_epoch = 0;
    let weights = dir.join("target/massive.safetensors");

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0f32;
        let mut steps = 0usize;

        for chunk in train.chunks(BATCH) {
            let (inputs, targets) = batch(chunk, device)?;

            let logits = forward(
                &inputs,
                &keys,
                &values,
            )?;

            let error =
                loss::cross_entropy(&logits, &targets)?;

            total_loss += error.to_scalar::<f32>()?;
            optimizer.backward_step(&error)?;
            steps += 1;
        }

        let score = accuracy(
            &validation,
            &keys,
            &values,
            device,
        )?;

        println!(
            "Epoch {epoch:02}/{EPOCHS} | loss {:.4} | validation {:.2}%",
            total_loss / steps as f32,
            score
        );

        if score > best {
            best = score;
            best_epoch = epoch;
            vars.save(&weights)?;
        }
    }

    vars.load(&weights)?;

    println!(
        "Best epoch: {best_epoch} | validation {best:.2}%"
    );

    println!(
        "FINAL TEST: {:.2}%",
        accuracy(&test, &keys, &values, device)?
    );

    let names: Vec<&str> =
        labels.keys().map(String::as_str).collect();

    std::fs::write(
        dir.join("target/massive-labels.json"),
        serde_json::to_vec_pretty(&names)?,
    )?;

    println!(
        "Model weights: {} bytes",
        std::fs::metadata(&weights)?.len()
    );

    Ok(())
}

fn predict(dir: &Path, device: &Device) -> Result<()> {
    let labels: Vec<String> = serde_json::from_slice(
        &std::fs::read(
            dir.join("target/massive-labels.json"),
        )?,
    )?;

    let (mut vars, keys, values) =
        setup(device, labels.len())?;

    vars.load(dir.join("target/massive.safetensors"))?;

    println!("Écris une phrase (vide pour quitter) :");

    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();

        if stdin.read_line(&mut line)? == 0
            || line.trim().is_empty()
        {
            break;
        }

        let inputs = Tensor::from_vec(
            encode(&line),
            (1, FEATURES),
            device,
        )?;

        let scores = candle_nn::ops::softmax(
            &forward(&inputs, &keys, &values)?,
            1,
        )?
        .to_vec2::<f32>()?;

        let mut ranking: Vec<usize> =
            (0..labels.len()).collect();

        ranking.sort_by(|a, b| {
            scores[0][*b].total_cmp(&scores[0][*a])
        });

        for index in ranking.into_iter().take(3) {
            println!(
                "  {:>6.2}%  {}",
                100.0 * scores[0][index],
                intent_names::display(&labels[index])
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let device =
        Device::new_cuda(0).context("CUDA unavailable")?;

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    match std::env::args().nth(1).as_deref() {
        Some("train") => train(dir, &device),
        Some("predict") => predict(dir, &device),
        _ => anyhow::bail!("Usage: massive [train|predict]"),
    }
}
