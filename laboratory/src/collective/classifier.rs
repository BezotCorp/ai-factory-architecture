use crate::instance::Instance;
use agent_collective::{Brain, FEATURES, Strategy};
use anyhow::Result;
use associative_circuits::encode;
use candle_core::{Device, Tensor};
use massive_intents::{Prediction, Predictor};
use std::path::Path;

struct Classifier {
    brain: Brain,
    labels: Vec<String>,
    strategy: Strategy,
}

impl Classifier {
    fn load(root: &Path, device: &Device, strategy: Strategy, id: Option<&str>) -> Result<Self> {
        let directory = Instance::resolve(root, "agent-collective", Some(strategy.name()), id)?;

        let labels: Vec<String> =
            serde_json::from_slice(&std::fs::read(directory.join("massive-labels.json"))?)?;

        let mut brain = Brain::new(device, labels.len())?;
        brain.load(Instance::weights(&directory)?)?;

        Ok(Self {
            brain,
            labels,
            strategy,
        })
    }
}

impl Predictor for Classifier {
    fn name(&self) -> &str {
        match self.strategy {
            Strategy::Focused => "agent-collective / focused",
            Strategy::Cooperative => "agent-collective / cooperative",
            Strategy::Local => "agent-collective / local",
        }
    }

    fn predict(&self, text: &str, limit: usize) -> Result<Vec<Prediction>> {
        let input = Tensor::from_vec(encode(text), (1, FEATURES), self.brain.device())?;

        let scores = self.brain.probabilities(&input)?;
        let mut indices: Vec<usize> = (0..self.labels.len()).collect();

        indices.sort_by(|a, b| scores[0][*b].total_cmp(&scores[0][*a]));

        Ok(indices
            .into_iter()
            .take(limit)
            .map(|i| Prediction {
                intent: massive_intents::display(&self.labels[i]).to_string(),
                score: scores[0][i],
            })
            .collect())
    }
}

pub(in crate::collective) fn compare(
    root: &Path,
    device: &Device,
    strategy: Strategy,
    id: Option<&str>,
) -> Result<()> {
    let classifier = Classifier::load(root, device, strategy, id)?;
    let suite = massive_intents::load_suite(root.join("tests/intent_cases.ron"))?;

    massive_intents::evaluate(&classifier, &suite)
}
