pub(crate) struct Regulator;

impl Regulator {
    pub(crate) fn weight(error: f32, average: f32) -> f64 {
        if average <= 0.0 {
            return 1.0;
        }

        ((error / average) as f64).clamp(0.5, 1.5)
    }
}
