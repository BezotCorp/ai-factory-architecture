use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Case {
    pub input: String,
    pub expected: Option<String>,
}
