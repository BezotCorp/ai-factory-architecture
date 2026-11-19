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
    let args: Vec<String> = std::env::args()
        .skip(2)
        .collect();

    if args.len() != 2 {
        bail!(
            "Usage: laboratory experimental [targeted|dual] [train|compare]"
        );
    }

    let (variant, lanes) = match args[0].as_str() {
        "targeted" => ("agent-targeted", 1),
        "dual" => ("agent-dual", 2),
        _ => bail!("Unknown experiment"),
    };

    match args[1].as_str() {
        "train" => training::train(
            root,
            device,
            variant,
            lanes,
        ),
        "compare" => classifier::compare(
            root,
            device,
            variant,
            lanes,
        ),
        _ => bail!("Expected train or compare"),
    }
}
