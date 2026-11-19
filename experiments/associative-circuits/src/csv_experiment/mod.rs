mod evaluation;
mod example;
mod load;
use anyhow::{Context, Result};
use associative_circuits::{forward, FEATURES};
use candle_core::{DType, Device};
use candle_nn::{loss, AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use evaluation::{accuracy, batch};
use example::Example;
use load::load;
use std::{collections::HashMap, path::Path};
const CIRCUITS: usize = 256;
const BATCH: usize = 128;
const EPOCHS: usize = 20;

pub fn run() -> Result<()> {
    let device =
        Device::new_cuda(0).context("CUDA unavailable: check NVIDIA driver and toolkit")?;

    println!("CUDA initialized");

    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let mut labels = HashMap::new();
    let train = load(&base.join("train.csv"), &mut labels, true)?;
    let test = load(&base.join("test.csv"), &mut labels, false)?;

    println!(
        "{} training examples, {} test examples, {} intents",
        train.len(),
        test.len(),
        labels.len()
    );

    let vars = VarMap::new();
    let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);

    let keys = builder.get_with_hints(
        (FEATURES, CIRCUITS),
        "associations",
        Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        },
    )?;

    let values = builder.get_with_hints(
        (CIRCUITS, labels.len()),
        "responses",
        Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        },
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
            epoch,
            EPOCHS,
            total_loss / steps as f32,
            score
        );
    }

    Ok(())
}
