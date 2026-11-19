use super::classifier::MassiveClassifier;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;

pub fn compare(root: &Path, device: &Device) -> Result<()> {
    let classifier = MassiveClassifier::load(root, device)?;
    let suite = massive_intents::load_suite(
        root.join("tests/intent_cases.ron")
    )?;

    massive_intents::evaluate(&classifier, &suite)
}
