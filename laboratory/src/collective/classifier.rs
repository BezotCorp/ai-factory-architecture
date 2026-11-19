use agent_collective::{Brain, Strategy, FEATURES};
use anyhow::Result;
use associative_circuits::encode;
use candle_core::{Device, Tensor};
use massive_intents::{Prediction, Predictor};
use std::path::Path;

pub(super) struct Classifier {
    brain: Brain,
    labels: Vec<String>,
    strategy: Strategy,
}

impl Classifier {
    fn load(
        root: &Path,
        device: &Device,
        strategy: Strategy,
    ) -> Result<Self> {
        let directory = root.join("artifacts").join(
            format!("agent-collective-{}", strategy.name())
        );

        let labels: Vec<String> = serde_json::from_slice(
            &std::fs::read(directory.join("massive-labels.json"))?
        )?;

        let mut brain = Brain::new(device, labels.len())?;
        brain.load(directory.join("massive.safetensors"))?;

        Ok(Self { brain, labels, strategy })
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
        let input = Tensor::from_vec(
            encode(text),
            (1, FEATURES),
            self.brain.device(),
        )?;

        let scores = self.brain.probabilities(&input)?;
        let mut indices: Vec<usize> = (0..self.labels.len()).collect();

        indices.sort_by(
            |a, b| scores[0][*b].total_cmp(&scores[0][*a])
        );

        Ok(indices.into_iter()
            .take(limit)
            .map(|i| Prediction {
                intent: massive_intents::display(
                    &self.labels[i]
                ).to_string(),
                score: scores[0][i],
            })
            .collect())
    }
}

pub(super) fn compare(
    root: &Path,
    device: &Device,
    strategy: Strategy,
) -> Result<()> {
    let classifier = Classifier::load(root, device, strategy)?;
    let suite = massive_intents::load_suite(
        root.join("tests/intent_cases.ron")
    )?;

    massive_intents::evaluate(&classifier, &suite)
}
