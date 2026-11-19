mod massive;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;

fn main() -> Result<()> {
    let device = if std::env::var_os("AI_FACTORY_CPU").is_some() {
        Device::Cpu
    } else {
        Device::new_cuda(0).context("CUDA unavailable")?
    };

    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Workspace root unavailable")?;

    match std::env::args().nth(1).as_deref() {
        Some("train") => massive::train(root, &device),
        Some("predict") => massive::predict(root, &device),
        Some("compare") => massive::compare(root, &device),
        _ => anyhow::bail!("Usage: laboratory [train|predict|compare]"),
    }
}
