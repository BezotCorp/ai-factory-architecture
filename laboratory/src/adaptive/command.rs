use crate::adaptive::{classifier, training};
use anyhow::{Result, bail};
use candle_core::Device;
use std::path::Path;

pub(crate) fn run_adaptive(root: &Path, device: &Device) -> Result<()> {
    let mut args = std::env::args().skip(2);
    let command = args.next().unwrap_or_default();

    let id = args.next();

    if command == "train" && id.is_some() {
        bail!("Train cannot select an existing instance");
    }

    if args.next().is_some() {
        bail!("Usage: laboratory adaptive [train|compare]");
    }

    match command.as_str() {
        "train" => training::train(root, device),
        "compare" => classifier::compare(root, device, id.as_deref()),
        _ => bail!("Usage: laboratory adaptive [train|compare]"),
    }
}
