use crate::{Predictor, Suite};
use anyhow::Result;
use std::{fs, path::Path};

pub fn load_suite(path: impl AsRef<Path>) -> Result<Suite> {
    Ok(ron::from_str(&fs::read_to_string(path)?)?)
}

pub fn evaluate(brain: &impl Predictor, suite: &Suite) -> Result<()> {
    let mut checked = 0usize;
    let mut correct = 0usize;

    println!("\nModel: {}", brain.name());

    for case in &suite.cases {
        let results = brain.predict(&case.input, 3)?;

        println!("\n> {}", case.input);

        for item in &results {
            println!("  {:>6.2}%  {}", item.score * 100.0, item.intent);
        }

        if let Some(expected) = &case.expected {
            checked += 1;
            let ok = results.first().is_some_and(|p| &p.intent == expected);
            correct += usize::from(ok);

            println!(
                "  expected: {expected} | {}",
                if ok { "OK" } else { "FAIL" }
            );
        }
    }

    if checked > 0 {
        println!("\nScored: {correct}/{checked} (only cases with expected intent)");
    }

    Ok(())
}
