use crate::Case;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Suite {
    pub cases: Vec<Case>,
}
