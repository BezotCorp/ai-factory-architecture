use anyhow::Result;
use associative_circuits::{encode, forward, FEATURES};
use candle_core::Tensor;

pub(super) struct MassiveClassifier {
    pub(super) keys: Tensor,
    pub(super) values: Tensor,
    pub(super) labels: Vec<String>,
}

impl massive_intents::Predictor for MassiveClassifier {
    fn name(&self) -> &str {
        "associative-circuits / MASSIVE (128)"
    }

    fn predict(&self, text: &str, limit: usize) -> Result<Vec<massive_intents::Prediction>> {
        let device = self.keys.device();
        let input = Tensor::from_vec(encode(text), (1, FEATURES), device)?;

        let scores = candle_nn::ops::softmax(&forward(&input, &self.keys, &self.values)?, 1)?
            .to_vec2::<f32>()?;

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
