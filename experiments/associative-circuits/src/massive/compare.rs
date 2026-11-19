use super::classifier::MassiveClassifier;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;

pub fn compare(dir: &Path, device: &Device) -> Result<()> {
    let labels: Vec<String> =
        serde_json::from_slice(&std::fs::read(dir.join("target/massive-labels.json"))?)?;

    let (mut vars, keys, values) =
        associative_circuits::setup(device, labels.len(), super::CIRCUITS)?;
    vars.load(dir.join("target/massive.safetensors"))?;

    let brain = MassiveClassifier {
        keys,
        values,
        labels,
    };
    let suite = massive_intents::load_suite(dir.join("../../tests/intent_cases.ron"))?;

    massive_intents::evaluate(&brain, &suite)
}
