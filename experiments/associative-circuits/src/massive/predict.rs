use anyhow::Result;
use associative_circuits::{encode, forward, FEATURES};
use candle_core::{Device, Tensor};
use std::{
    io::{self, Write},
    path::Path,
};

pub fn predict(dir: &Path, device: &Device) -> Result<()> {
    let labels: Vec<String> =
        serde_json::from_slice(&std::fs::read(dir.join("target/massive-labels.json"))?)?;

    let (mut vars, keys, values) =
        associative_circuits::setup(device, labels.len(), super::CIRCUITS)?;

    vars.load(dir.join("target/massive.safetensors"))?;

    println!("Écris une phrase (vide pour quitter) :");

    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();

        if stdin.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }

        let inputs = Tensor::from_vec(encode(&line), (1, FEATURES), device)?;

        let scores =
            candle_nn::ops::softmax(&forward(&inputs, &keys, &values)?, 1)?.to_vec2::<f32>()?;

        let mut ranking: Vec<usize> = (0..labels.len()).collect();

        ranking.sort_by(|a, b| scores[0][*b].total_cmp(&scores[0][*a]));

        for index in ranking.into_iter().take(3) {
            println!(
                "  {:>6.2}%  {}",
                100.0 * scores[0][index],
                massive_intents::display(&labels[index])
            );
        }
    }

    Ok(())
}
