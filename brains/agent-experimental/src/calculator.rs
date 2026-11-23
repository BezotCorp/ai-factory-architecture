use candle_core::{DType, Tensor};

pub(crate) struct Calculator;

impl Calculator {
    pub(crate) fn error(logits: &Tensor, targets: &Tensor) -> candle_core::Result<f32> {
        candle_nn::loss::cross_entropy(logits, targets)?.to_scalar::<f32>()
    }

    pub(crate) fn expected(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
        let classes = logits.dim(1)?;
        let labels = targets.to_vec1::<u32>()?;
        let mut values = vec![0f32; labels.len() * classes];

        for (row, label) in labels.into_iter().enumerate() {
            let column = label as usize;
            if column >= classes {
                candle_core::bail!("Target outside class range");
            }
            values[row * classes + column] = 1.0;
        }

        Tensor::from_vec(values, (targets.dim(0)?, classes), logits.device())
    }

    pub(crate) fn local_errors(logits: &Tensor, expected: &Tensor) -> candle_core::Result<Tensor> {
        let probability = candle_nn::ops::softmax(logits, 1)?;
        let correct_probability = probability.mul(expected)?.sum(1)?;

        correct_probability
            .clamp(1e-8, 1.0)?
            .log()?
            .affine(-1.0, 0.0)
    }

    pub(crate) fn residual(logits: &Tensor, expected: &Tensor) -> candle_core::Result<Tensor> {
        expected - &candle_nn::ops::softmax(logits, 1)?
    }

    // Mesure expérimentale : quel effet une variation
    // positive ou négative d'un neurone produit-elle
    // sur l'erreur de chaque exemple ?
    //
    // Aucun appel à backward() ni au moteur autograd.
    pub(crate) fn direction(
        logits: &Tensor,
        expected: &Tensor,
        outgoing: &Tensor,
        total_agents: usize,
    ) -> candle_core::Result<Tensor> {
        const STEP: f64 = 0.1;

        let change = outgoing.affine(STEP / total_agents as f64, 0.0)?;

        let plus = logits.broadcast_add(&change)?;
        let minus = logits.broadcast_sub(&change)?;

        let plus_error = Self::local_errors(&plus, expected)?;
        let minus_error = Self::local_errors(&minus, expected)?;

        (&minus_error - &plus_error)?
            .affine(1.0 / (2.0 * STEP), 0.0)?
            .unsqueeze(1)?
            .to_dtype(DType::F32)
    }
}
