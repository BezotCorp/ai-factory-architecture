use crate::{Dataset, Row};
use anyhow::{Context, Result, bail};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;

const BASE: &str =
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/refs%2Fconvert%2Fparquet/fr-FR";

fn load_split(client: &reqwest::blocking::Client, split: &str) -> Result<Vec<Row>> {
    let url = format!("{BASE}/{split}/0000.parquet");

    println!("Loading {split} into RAM...");

    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("Cannot download {split}"))?
        .error_for_status()?;

    let bytes = response.bytes()?;
    let reader = SerializedFileReader::new(bytes)?;

    let mut examples = Vec::new();

    for record in reader.get_row_iter(None)? {
        let record = record?;
        let mut text = None;
        let mut intent = None;

        for (name, field) in record.get_column_iter() {
            match name.as_str() {
                "utt" => {
                    if let Field::Str(value) = field {
                        text = Some(value.clone());
                    }
                }
                "intent" => {
                    intent = match field {
                        Field::Str(value) => Some(value.clone()),
                        Field::Int(value) => Some(value.to_string()),
                        Field::Long(value) => Some(value.to_string()),
                        _ => None,
                    };
                }
                _ => {}
            }
        }

        examples.push(Row {
            text: text.context("Missing utterance")?,
            intent: intent.context("Missing intent")?,
        });
    }

    if examples.is_empty() {
        bail!("Empty dataset split: {split}");
    }

    println!("{split}: {} examples", examples.len());
    Ok(examples)
}

pub fn load() -> Result<Dataset> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("ai-factory-architecture/0.1")
        .build()?;

    Ok(Dataset {
        train: load_split(&client, "train")?,
        validation: load_split(&client, "validation")?,
        test: load_split(&client, "test")?,
    })
}
