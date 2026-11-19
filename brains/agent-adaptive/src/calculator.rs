use candle_core::Tensor;

pub(super) struct Calculator;

impl Calculator {
    pub(super) fn error(
        logits: &Tensor,
        targets: &Tensor,
    ) -> candle_core::Result<f32> {
        candle_nn::loss::cross_entropy(logits, targets)?
            .to_scalar::<f32>()
    }

    pub(super) fn signal(
        logits: &Tensor,
        targets: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let classes = logits.dim(1)?;
        let labels = targets.to_vec1::<u32>()?;
        let mut expected = vec![0f32; labels.len() * classes];

        for (row, label) in labels.iter().enumerate() {
            let index = *label as usize;
            if index >= classes {
                candle_core::bail!("Target outside class range");
            }
            expected[row * classes + index] = 1.0;
        }

        let expected = Tensor::from_vec(
            expected,
            (labels.len(), classes),
            logits.device(),
        )?;

        let probabilities =
            candle_nn::ops::softmax(logits, 1)?;

        &expected - &probabilities
    }

    pub(super) fn reward(before: f32, after: f32) -> f32 {
        if before <= 0.0 {
            return 0.0;
        }

        ((before - after) / before).clamp(-1.0, 1.0)
    }
}
