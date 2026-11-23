use crate::Strategy;
use crate::regulator::Regulator;
use anyhow::Result;
use candle_core::Tensor;

pub(crate) struct Teacher;

impl Teacher {
    pub(crate) fn lesson(losses: &[Tensor], strategy: Strategy) -> Result<Tensor> {
        let errors: Vec<f32> = losses
            .iter()
            .map(|loss| loss.to_scalar::<f32>())
            .collect::<candle_core::Result<_>>()?;

        match strategy {
            Strategy::Local => {
                anyhow::bail!("Local teaching uses LocalTeacher")
            }
            Strategy::Focused => {
                let index = errors
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .ok_or_else(|| anyhow::anyhow!("No experts"))?
                    .0;

                Ok(losses[index].clone())
            }

            Strategy::Cooperative => {
                let average = errors.iter().sum::<f32>() / errors.len() as f32;

                let mut total = losses[0].affine(Regulator::weight(errors[0], average), 0.0)?;

                for (loss, error) in losses.iter().zip(&errors).skip(1) {
                    let part = loss.affine(Regulator::weight(*error, average), 0.0)?;
                    total = (&total + &part)?;
                }

                Ok(total.affine(1.0 / losses.len() as f64, 0.0)?)
            }
        }
    }
}
