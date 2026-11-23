use crate::instance::Instance;
use anyhow::Result;
use associative_circuits::{Brain, FEATURES, encode};
use candle_core::{Device, Tensor};
use std::path::Path;

pub(in crate::massive) struct MassiveClassifier {
    brain: Brain,
    labels: Vec<String>,
}

impl MassiveClassifier {
    pub(in crate::massive) fn load(root: &Path, device: &Device) -> Result<Self> {
        Self::load_files(&root.join("artifacts/associative-circuits"), device)
    }

    pub(in crate::massive) fn load_instance(
        root: &Path,
        device: &Device,
        id: Option<&str>,
    ) -> Result<Self> {
        let directory = Instance::resolve(root, "associative-circuits", None, id)?;

        Self::load_files(&directory, device)
    }

    pub(in crate::massive) fn load_files(base: &Path, device: &Device) -> Result<Self> {
        let labels: Vec<String> =
            serde_json::from_slice(&std::fs::read(base.join("massive-labels.json"))?)?;

        let mut brain = Brain::new(device, labels.len(), crate::massive::config::CIRCUITS)?;
        brain.load(Instance::weights(base)?)?;

        Ok(Self { brain, labels })
    }
}

impl massive_intents::Predictor for MassiveClassifier {
    fn name(&self) -> &str {
        "associative-circuits / MASSIVE (128)"
    }

    fn predict(&self, text: &str, limit: usize) -> Result<Vec<massive_intents::Prediction>> {
        let input = Tensor::from_vec(encode(text), (1, FEATURES), self.brain.device())?;

        let scores = self.brain.probabilities(&input)?;

        let mut indices: Vec<usize> = (0..self.labels.len()).collect();
        indices.sort_by(|a, b| scores[0][*b].total_cmp(&scores[0][*a]));

        Ok(indices
            .into_iter()
            .take(limit)
            .map(|i| massive_intents::Prediction {
                intent: massive_intents::display(&self.labels[i]).to_string(),
                score: scores[0][i],
            })
            .collect())
    }
}
