use super::classifier::MassiveClassifier;
use anyhow::Result;
use candle_core::Device;
use massive_intents::Predictor;
use std::{
    io::{self, Write},
    path::Path,
};

pub fn predict(root: &Path, device: &Device) -> Result<()> {
    let classifier = MassiveClassifier::load(root, device)?;

    println!("Écris une phrase (vide pour quitter) :");

    let stdin = io::stdin();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();

        if stdin.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }

        for result in classifier.predict(&line, 3)? {
            println!(
                "  {:>6.2}%  {}",
                result.score * 100.0,
                result.intent
            );
        }
    }

    Ok(())
}
