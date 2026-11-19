mod massive;
mod adaptive;
mod experimental;
mod collective;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Workspace root unavailable")?;

    if args.first().map(String::as_str) == Some("publish") {
        return massive::publish(root, &args[1..]);
    }

    let device = if std::env::var_os("AI_FACTORY_CPU").is_some() {
        Device::Cpu
    } else {
        Device::new_cuda(0).context("CUDA unavailable")?
    };

    match args.first().map(String::as_str) {
        Some("collective") => collective::run(root, &device),
        Some("adaptive") => adaptive::run(root, &device),
        Some("experimental") => experimental::run(root, &device),
        Some("train") => massive::train(root, &device),
        Some("predict") => massive::predict(root, &device),
        Some("compare") => massive::compare(root, &device),
        _ => anyhow::bail!("Usage: laboratory [train|predict|compare|publish]"),
    }
}
