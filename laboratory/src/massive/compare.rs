use crate::massive::classifier::MassiveClassifier;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;

pub(crate) fn compare_massive(root: &Path, device: &Device, id: Option<&str>) -> Result<()> {
    let classifier = MassiveClassifier::load_instance(root, device, id)?;
    let suite = massive_intents::load_suite(root.join("tests/intent_cases.ron"))?;

    massive_intents::evaluate(&classifier, &suite)
}
