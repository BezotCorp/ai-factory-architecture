use crate::collective::{classifier, training_data};
use agent_collective::Strategy;
use anyhow::{Result, bail};
use candle_core::Device;
use std::path::Path;

pub(crate) fn run_collective(root: &Path, device: &Device) -> Result<()> {
    let mut args = std::env::args().skip(2);
    let command = args.next().unwrap_or_default();
    let variant = args.next().unwrap_or_default();

    let Some(strategy) = Strategy::parse(&variant) else {
        bail!("Usage: collective [train|compare] [focused|cooperative]");
    };

    let id = args.next();

    if command == "train" && id.is_some() {
        bail!("Train cannot select an existing instance");
    }

    if args.next().is_some() {
        bail!("Unexpected argument");
    }

    match command.as_str() {
        "train" => training_data::train(root, device, strategy),
        "compare" => classifier::compare(root, device, strategy, id.as_deref()),
        _ => bail!("Unknown collective command"),
    }
}
