use crate::experimental::{classifier, training};
use anyhow::{Result, bail};
use candle_core::Device;
use std::path::Path;

pub(crate) fn run_experimental(root: &Path, device: &Device) -> Result<()> {
    let args: Vec<String> = std::env::args().skip(2).collect();

    if !(2..=3).contains(&args.len()) {
        bail!("Usage: laboratory experimental [targeted|dual] [train|compare]");
    }

    let (variant, lanes) = match args[0].as_str() {
        "targeted" => ("agent-targeted", 1),
        "dual" => ("agent-dual", 2),
        _ => bail!("Unknown experiment"),
    };

    match args[1].as_str() {
        "train" if args.len() == 2 => training::train(root, device, variant, lanes),
        "compare" => classifier::compare(
            root,
            device,
            variant,
            lanes,
            args.get(2).map(String::as_str),
        ),
        _ => bail!("Expected train or compare"),
    }
}
