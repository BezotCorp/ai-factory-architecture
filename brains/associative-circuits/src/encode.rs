use crate::FEATURES;

pub fn encode(text: &str) -> Vec<f32> {
    let lower = text.to_lowercase();
    let mut vector = vec![0f32; FEATURES];

    for gram in lower.as_bytes().windows(3) {
        let mut hash = 2166136261u32;

        for byte in gram {
            hash = (hash ^ *byte as u32).wrapping_mul(16777619);
        }

        vector[(hash as usize) % FEATURES] += 1.0;
    }

    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }

    vector
}
