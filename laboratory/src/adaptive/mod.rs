mod adaptive_encoding;
mod classifier;
mod example;
mod training;

use anyhow::{bail, Result};
use candle_core::Device;
use std::path::Path;

pub(super) const BATCH: usize = 128;
pub(super) const EPOCHS: usize = 20;

pub fn run(root: &Path, device: &Device) -> Result<()> {
    let mut args = std::env::args().skip(2);
    let command = args.next().unwrap_or_default();

    if args.next().is_some() {
        bail!("Usage: laboratory adaptive [train|compare]");
    }

    match command.as_str() {
        "train" => training::train(root, device),
        "compare" => classifier::compare(root, device),
        _ => bail!("Usage: laboratory adaptive [train|compare]"),
    }
}
