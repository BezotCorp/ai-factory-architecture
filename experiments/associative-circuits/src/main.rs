
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

const FEATURES: usize = 1024;
const CIRCUITS: usize = 256;
const BATCH: usize = 128;
const EPOCHS: usize = 20;

struct Example {
    features: Vec<f32>,
    label: u32,
}

fn encode(text: &str) -> Vec<f32> {
    let mut vector = vec![0f32; FEATURES];
    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();

    for gram in bytes.windows(3) {
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

fn load(
    path: &Path,
    labels: &mut HashMap<String, u32>,
    allow_new: bool,
) -> Result<Vec<Example>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut examples = Vec::new();

    for row in reader.records() {
        let row = row?;
        let text = row.get(0).context("Missing text")?;
        let name = row.get(1).context("Missing label")?;

        let id = match labels.get(name) {
            Some(id) => *id,
            None if allow_new => {
                let id = labels.len() as u32;
                labels.insert(name.to_string(), id);
                id
            }
            None => anyhow::bail!("Unknown label: {name}"),
        };

        examples.push(Example {
            features: encode(text),
            label: id,
        });
    }

    Ok(examples)
}

fn batch(data: &[Example], device: &Device) -> Result<(Tensor, Tensor)> {
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

fn forward(
    inputs: &Tensor,
    keys: &Tensor,
    values: &Tensor,
) -> candle_core::Result<Tensor> {
    let activation = inputs.matmul(keys)?.relu()?;
    activation.matmul(values)
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
        let predicted = forward(&inputs, keys, values)?
            .argmax(1)?
            .to_vec1::<u32>()?;

        correct += predicted.iter()
            .zip(chunk.iter())
            .filter(|(prediction, example)| **prediction == example.label)
            .count();
    }

    Ok(100.0 * correct as f32 / data.len() as f32)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)
        .context("CUDA unavailable: check NVIDIA driver and toolkit")?;

    println!("CUDA initialized");

    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut labels = HashMap::new();
    let train = load(&base.join("train.csv"), &mut labels, true)?;
    let test = load(&base.join("test.csv"), &mut labels, false)?;

    println!(
        "{} training examples, {} test examples, {} intents",
        train.len(), test.len(), labels.len()
    );

    let vars = VarMap::new();
    let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let keys = builder.get_with_hints(
        (FEATURES, CIRCUITS),
        "associations",
        Init::Randn { mean: 0.0, stdev: 0.05 },
    )?;

    let values = builder.get_with_hints(
        (CIRCUITS, labels.len()),
        "responses",
        Init::Randn { mean: 0.0, stdev: 0.05 },
    )?;

    let mut optimizer = AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: 0.003,
            ..Default::default()
        },
    )?;

    println!("Training {} associative circuits...", CIRCUITS);

    for epoch in 1..=EPOCHS {
        let mut total_loss = 0f32;
        let mut steps = 0usize;

        for chunk in train.chunks(BATCH) {
            let (inputs, targets) = batch(chunk, &device)?;
            let logits = forward(&inputs, &keys, &values)?;
            let error = loss::cross_entropy(&logits, &targets)?;

            total_loss += error.to_scalar::<f32>()?;
            optimizer.backward_step(&error)?;
            steps += 1;
        }

        let score = accuracy(&test, &keys, &values, &device)?;

        println!(
            "Epoch {:02}/{} | loss {:.4} | test accuracy {:.2}%",
            epoch, EPOCHS, total_loss / steps as f32, score
        );
    }

    Ok(())
}
