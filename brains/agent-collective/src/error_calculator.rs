use candle_core::Tensor;

pub(super) struct ErrorCalculator;

impl ErrorCalculator {

    pub(super) fn signal(
        logits: &Tensor,
        targets: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let classes = logits.dim(1)?;
        let labels = targets.to_vec1::<u32>()?;
        let mut expected = vec![0f32; labels.len() * classes];

        for (row, &label) in labels.iter().enumerate() {
            let index = label as usize;
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

        let probabilities = candle_nn::ops::softmax(logits, 1)?;

        // Correction souhaitée : cible moins prédiction.
        &expected - &probabilities
    }

    pub(super) fn measure(
        logits: &Tensor,
        targets: &Tensor,
    ) -> candle_core::Result<Tensor> {
        candle_nn::loss::cross_entropy(logits, targets)
    }
}
