use crate::massive::{classifier::MassiveClassifier, config::CIRCUITS};
use anyhow::{Context, Result, bail};
use associative_circuits::FEATURES;
use candle_core::Device;
use std::{fs, path::Path, process::Command};

const MODEL_CARD: &str = r#"---
language:
- fr
tags:
- intent-classification
- associative-circuits
datasets:
- AmazonScience/massive
---

# Associative circuits for French intent classification

Experimental intent classifier trained on the French (fr-FR) split of
[Amazon MASSIVE](https://huggingface.co/datasets/AmazonScience/massive).
It is part of the AI Factory Architecture research project. This is a custom
Rust/Candle model, not a Transformers model.

## Architecture and input

Text is lowercased and encoded as hashed UTF-8 byte trigrams using FNV-1a,
with {features} features and L2 normalization. The model computes
`ReLU(input × associations) × responses`, with {circuits} associative
circuits. A softmax turns its logits into class scores.

`massive.safetensors` contains the `associations` and `responses` tensors.
`massive-labels.json` maps output positions to MASSIVE intent IDs; the two
files must be kept together. The application maps numeric IDs to intent names.

## Usage

Use the AI Factory Architecture Rust project to run this model. Place both
files in `artifacts/associative-circuits/` at the workspace root, then run:

```sh
AI_FACTORY_CPU=1 cargo run --no-default-features -p laboratory -- compare
# Or use `predict` instead of `compare` for interactive inference.
```

The optional CPU mode above builds without CUDA. See the source project for
its complete execution and evaluation code.

## Limitations

This is an experimental classifier, not a general-purpose language model.
Predictions outside the training domain can be unreliable. No evaluation
metrics are included here because they are not stored with the model files.
"#;

fn prepare_massive(root: &Path, destination: &Path) -> Result<()> {
    let source = root.join("artifacts/associative-circuits");
    // Verify that the exact existing weights and labels can be loaded together.
    MassiveClassifier::load(root, &Device::Cpu).context("Cannot load model and labels")?;
    fs::create_dir(destination)
        .with_context(|| format!("Cannot create new directory {}", destination.display()))?;
    for name in ["massive.safetensors", "massive-labels.json"] {
        fs::copy(source.join(name), destination.join(name))
            .with_context(|| format!("Cannot copy {name}"))?;
    }
    let card = MODEL_CARD
        .replace("{features}", &FEATURES.to_string())
        .replace("{circuits}", &CIRCUITS.to_string());
    fs::write(destination.join("README.md"), card)?;
    Ok(())
}

fn upload(repo: &str, directory: &Path) -> Result<()> {
    if repo.starts_with('-') || repo.split('/').count() != 2 || repo.split('/').any(str::is_empty) {
        bail!("Expected repository in the form namespace/model");
    }
    MassiveClassifier::load_files(directory, &Device::Cpu)
        .context("Cannot load publication weights and labels")?;
    if !directory.join("README.md").is_file() {
        bail!("Publication directory must contain README.md");
    }
    let status = Command::new("hf")
        .arg("upload")
        .arg(repo)
        .arg(directory)
        .arg(".")
        .status()
        .context("Cannot run hf; install the Hugging Face CLI and run hf auth login")?;
    if !status.success() {
        bail!("Hugging Face upload failed: {status}");
    }
    Ok(())
}

pub(crate) fn run_massive(root: &Path, args: &[String]) -> Result<()> {
    match args {
        [option, path] if option == "--prepare-only" => {
            let destination = Path::new(path);
            prepare_massive(root, destination)?;
            println!("Publication files prepared in {}", destination.display());
            Ok(())
        }
        [repo, option, path] if option == "--from" => upload(repo, Path::new(path)),
        [repo] => {
            let temporary = tempfile::tempdir().context("Cannot create temporary directory")?;
            let destination = temporary.path().join("model");
            prepare_massive(root, &destination)?;
            upload(repo, &destination)
        }
        _ => bail!(
            "Usage: laboratory publish <namespace/model> [--from <directory>] | publish --prepare-only <new-directory>"
        ),
    }
}
