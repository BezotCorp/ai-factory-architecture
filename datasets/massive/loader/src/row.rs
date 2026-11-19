use serde::Deserialize;

#[derive(Deserialize)]
pub struct Row {
    pub text: String,
    pub intent: String,
}
