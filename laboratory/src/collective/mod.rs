mod classifier;
mod example;
mod training_data;

use agent_collective::Strategy;
use anyhow::{bail, Result};
use candle_core::Device;
use std::path::Path;

const BATCH: usize = 128;
const EPOCHS: usize = 20;

pub fn run(root: &Path, device: &Device) -> Result<()> {
    let mut args = std::env::args().skip(2);
    let command = args.next().unwrap_or_default();
    let variant = args.next().unwrap_or_default();

    let Some(strategy) = Strategy::parse(&variant) else {
        bail!("Usage: collective [train|compare] [focused|cooperative]");
    };

    if args.next().is_some() {
        bail!("Unexpected argument");
    }

    match command.as_str() {
        "train" => training_data::train(root, device, strategy),
        "compare" => classifier::compare(root, device, strategy),
        _ => bail!("Unknown collective command"),
    }
}
