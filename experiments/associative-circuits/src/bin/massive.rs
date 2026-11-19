#[path = "../massive/mod.rs"]
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
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    match std::env::args().nth(1).as_deref() {
        Some("train") => massive::train(&dir, &device),
        Some("predict") => massive::predict(&dir, &device),
        Some("compare") => massive::compare(&dir, &device),
        _ => anyhow::bail!("Usage: massive [train|predict|compare]"),
    }
}
