use super::Row;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Dataset {
    pub train: Vec<Row>,
    pub validation: Vec<Row>,
    pub test: Vec<Row>,
}
