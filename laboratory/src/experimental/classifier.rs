use crate::instance::Instance;
use std::path::Path;

use agent_experimental::{Brain, FEATURES};
use anyhow::Result;
use candle_core::{Device, Tensor};
use massive_intents::{Prediction, Predictor};

use crate::experimental::adaptive_encoding::encode;

struct Classifier {
    brain: Brain,
    labels: Vec<String>,
}

impl Classifier {
    fn load(
        root: &Path,
        device: &Device,
        variant: &str,
        lanes: usize,
        id: Option<&str>,
    ) -> Result<Self> {
        let dir = Instance::resolve(
            root,
            "agent-experimental",
            Some(variant.strip_prefix("agent-").unwrap_or(variant)),
            id,
        )?;

        let labels: Vec<String> =
            serde_json::from_slice(&std::fs::read(dir.join("massive-labels.json"))?)?;

        let mut brain = Brain::new(device, labels.len(), lanes)?;
        brain.load(Instance::weights(&dir)?)?;

        Ok(Self { brain, labels })
    }
}

impl Predictor for Classifier {
    fn name(&self) -> &str {
        "agent-experimental / targeted teaching"
    }

    fn predict(&self, text: &str, limit: usize) -> Result<Vec<Prediction>> {
        let input = Tensor::from_vec(encode(text), (1, FEATURES), self.brain.device())?;

        let scores = self.brain.probabilities(&input)?;
        let mut indices: Vec<usize> = (0..self.labels.len()).collect();

        indices.sort_by(|a, b| scores[0][*b].total_cmp(&scores[0][*a]));

        Ok(indices
            .into_iter()
            .take(limit)
            .map(|index| Prediction {
                intent: massive_intents::display(&self.labels[index]).to_string(),
                score: scores[0][index],
            })
            .collect())
    }
}

pub(in crate::experimental) fn compare(
    root: &Path,
    device: &Device,
    variant: &str,
    lanes: usize,
    id: Option<&str>,
) -> Result<()> {
    let classifier = Classifier::load(root, device, variant, lanes, id)?;
    let suite = massive_intents::load_suite(root.join("tests/intent_cases.ron"))?;
    massive_intents::evaluate(&classifier, &suite)
}
