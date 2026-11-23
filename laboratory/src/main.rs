mod adaptive;
mod collective;
mod experimental;
mod instance;
mod massive;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Workspace root unavailable")?;

    if args.first().map(String::as_str) == Some("publish") {
        return instance::Instance::publish(root, &args[1..]);
    }

    if args.first().map(String::as_str) == Some("export") {
        if args.len() != 3 {
            anyhow::bail!("Usage: laboratory export <uuid> <destination>");
        }
        return instance::Instance::export(root, &args[1], Path::new(&args[2]));
    }

    if args.first().map(String::as_str) == Some("import") {
        if args.len() != 2 {
            anyhow::bail!("Usage: laboratory import <directory>");
        }
        return instance::Instance::import(root, Path::new(&args[1]));
    }

    let device = if std::env::var_os("AI_FACTORY_CPU").is_some() {
        Device::Cpu
    } else {
        Device::new_cuda(0).context("CUDA unavailable")?
    };

    match args.first().map(String::as_str) {
        Some("collective") => collective::run_collective(root, &device),
        Some("adaptive") => adaptive::run_adaptive(root, &device),
        Some("experimental") => experimental::run_experimental(root, &device),
        Some("train") => massive::train_massive(root, &device),
        Some("predict") => massive::predict_massive(root, &device),
        Some("compare") => massive::compare_massive(root, &device, args.get(1).map(String::as_str)),
        _ => anyhow::bail!("Usage: laboratory [train|predict|compare|publish]"),
    }
}
