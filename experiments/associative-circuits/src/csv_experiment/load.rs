use super::Example;
use anyhow::{Context, Result};
use associative_circuits::encode;
use std::{collections::HashMap, path::Path};

pub(super) fn load(
    path: &Path,
    labels: &mut HashMap<String, u32>,
    allow_new: bool,
) -> Result<Vec<Example>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut examples = Vec::new();

    for row in reader.records() {
        let row = row?;
        let text = row.get(0).context("Missing text")?;
        let name = row.get(1).context("Missing label")?;

        let id = match labels.get(name) {
            Some(id) => *id,
            None if allow_new => {
                let id = labels.len() as u32;
                labels.insert(name.to_string(), id);
                id
            }
            None => anyhow::bail!("Unknown label: {name}"),
        };

        examples.push(Example {
            features: encode(text),
            label: id,
        });
    }

    Ok(examples)
}
