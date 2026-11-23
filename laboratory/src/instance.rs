use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

use crate::massive;

pub(crate) struct Instance {
    directory: PathBuf,
    manifest: Value,
}

impl Instance {
    pub(crate) fn new(
        root: &Path,
        family: &str,
        variant: Option<&str>,
        configuration: Value,
    ) -> Result<Self> {
        Self::create(root, family, variant, configuration)
    }

    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    fn verify_name(name: &str) -> Result<()> {
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            bail!("Invalid brain type or variant: {name}");
        }
        Ok(())
    }

    fn parent(root: &Path, family: &str, variant: Option<&str>) -> Result<PathBuf> {
        Self::verify_name(family)?;

        let mut path = root.join("artifacts").join(family);

        if let Some(name) = variant {
            Self::verify_name(name)?;
            path = path.join(name);
        }

        Ok(path)
    }

    pub fn create(
        root: &Path,
        family: &str,
        variant: Option<&str>,
        configuration: Value,
    ) -> Result<Self> {
        let parent = Self::parent(root, family, variant)?;
        fs::create_dir_all(&parent)?;

        let id = Uuid::now_v7();
        let directory = parent.join(id.to_string());

        fs::create_dir(&directory)?;
        fs::create_dir(directory.join("checkpoints"))?;

        let created_at_unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let git_commit = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(root)
            .output()
            .ok()
            .filter(|result| result.status.success())
            .map(|result| String::from_utf8_lossy(&result.stdout).trim().to_owned());

        let mut instance = Self {
            directory,
            manifest: json!({
                "id": id.to_string(),
                "family": family,
                "variant": variant,
                "created_at_unix": created_at_unix,
                "git_commit": git_commit,
                "dataset": {
                    "name": "AmazonScience/MASSIVE",
                    "language": "fr-FR"
                },
                "status": "training",
                "configuration": configuration,
                "checkpoints": [],
                "best_checkpoint": null,
                "metrics": null,
                "sources": "source",
                "resume_state_complete": false,
                "deterministic_replay_guaranteed": false
            }),
        };

        instance.snapshot(root)?;
        instance.write_manifest()?;

        let card = format!(
            "---\n\
             language:\n\
             - fr\n\
             tags:\n\
             - intent-classification\n\
             - custom-rust-model\n\
             ---\n\n\
             # AI Factory: {family}\n\n\
             Instance UUID: `{id}`\n\n\
             Custom Rust/Candle architecture.\n\
             The archived `source/` directory contains the implementation.\n\
             See `manifest.json` for configuration and checkpoints.\n\
             Exact training continuation and deterministic replay are not guaranteed.\n"
        );

        fs::write(instance.directory.join("README.md"), card)?;

        println!("New AI: {}", instance.directory.display());
        Ok(instance)
    }

    fn snapshot(&self, root: &Path) -> Result<()> {
        let destination = self.directory.join("source");

        for folder in ["brains", "datasets", "laboratory", "tests"] {
            let source = root.join(folder);

            if source.is_dir() {
                Self::copy_sources(&source, &destination.join(folder))?;
            }
        }

        for name in ["Cargo.toml", "Cargo.lock"] {
            let source = root.join(name);

            if source.is_file() {
                fs::create_dir_all(&destination)?;
                fs::copy(&source, destination.join(name))?;
            }
        }

        Ok(())
    }

    fn copy_sources(from: &Path, to: &Path) -> Result<()> {
        for entry in fs::read_dir(from)? {
            let entry = entry?;
            let kind = entry.file_type()?;

            if kind.is_symlink() {
                continue;
            }

            let source = entry.path();
            let destination = to.join(entry.file_name());

            if kind.is_dir() {
                Self::copy_sources(&source, &destination)?;
            } else if kind.is_file() {
                let ext = source
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("");

                if matches!(ext, "rs" | "toml" | "ron" | "md" | "json") {
                    fs::create_dir_all(to)?;
                    fs::copy(&source, &destination)?;
                }
            }
        }

        Ok(())
    }

    fn write_manifest(&mut self) -> Result<()> {
        let temporary = self.directory.join("manifest.json.tmp");

        fs::write(&temporary, serde_json::to_vec_pretty(&self.manifest)?)?;

        fs::rename(&temporary, self.directory.join("manifest.json"))?;

        Ok(())
    }

    pub fn checkpoint(&self, epoch: usize) -> PathBuf {
        self.directory
            .join("checkpoints")
            .join(format!("epoch_{epoch:04}.safetensors"))
    }

    pub fn labels(&self) -> PathBuf {
        self.directory.join("massive-labels.json")
    }

    fn digest(file: &Path) -> Result<String> {
        let mut reader = fs::File::open(file)?;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 65536];

        loop {
            let count = reader.read(&mut buffer)?;

            if count == 0 {
                break;
            }

            digest.update(&buffer[..count]);
        }

        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn record(&mut self, epoch: usize, file: &Path, validation: f32) -> Result<()> {
        let relative = file
            .strip_prefix(&self.directory)?
            .to_string_lossy()
            .into_owned();

        let bytes = fs::metadata(file)?.len();
        let sha256 = Self::digest(file)?;

        self.manifest["checkpoints"]
            .as_array_mut()
            .context("Invalid checkpoints")?
            .push(json!({
                "epoch": epoch,
                "file": relative,
                "validation_accuracy": validation,
                "bytes": bytes,
                "sha256": sha256
            }));

        self.manifest["best_checkpoint"] = json!(relative);
        self.write_manifest()
    }

    pub fn finish(&mut self, epoch: usize, validation: f32, test: f32) -> Result<()> {
        if self.manifest["best_checkpoint"].is_null() || !self.labels().is_file() {
            bail!("Incomplete training instance");
        }

        self.manifest["status"] = json!("completed");
        self.manifest["metrics"] = json!({
            "best_epoch": epoch,
            "validation_accuracy": validation,
            "test_accuracy": test
        });

        self.write_manifest()?;

        println!("Stored AI: {}", self.directory.display());
        Ok(())
    }

    fn manifest(path: &Path) -> Result<Value> {
        Ok(serde_json::from_slice(&fs::read(
            path.join("manifest.json"),
        )?)?)
    }

    pub fn weights(directory: &Path) -> Result<PathBuf> {
        if !directory.join("manifest.json").exists() {
            return Ok(directory.join("massive.safetensors"));
        }

        let manifest = Self::manifest(directory)?;

        let name = manifest["best_checkpoint"]
            .as_str()
            .context("Missing best checkpoint")?;

        let relative = Path::new(name);

        if relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("Invalid checkpoint path");
        }

        let expected = manifest["checkpoints"]
            .as_array()
            .context("Invalid checkpoints")?
            .iter()
            .find(|item| item["file"] == name)
            .context("Unregistered checkpoint")?;

        let path = directory.join(relative);

        if Self::digest(&path)? != expected["sha256"].as_str().context("Missing SHA256")? {
            bail!("Checkpoint SHA256 mismatch");
        }

        Ok(path)
    }

    pub fn resolve(
        root: &Path,
        family: &str,
        variant: Option<&str>,
        id: Option<&str>,
    ) -> Result<PathBuf> {
        let parent = Self::parent(root, family, variant)?;

        if let Some(id) = id {
            let id = Uuid::parse_str(id)?.to_string();
            let path = parent.join(&id);
            let manifest = Self::manifest(&path)?;

            if manifest["id"] != id
                || manifest["family"] != family
                || manifest["variant"] != json!(variant)
            {
                bail!("Instance identity/type mismatch");
            }

            if manifest["status"] != "completed" {
                bail!("Instance not completed");
            }

            return Ok(path);
        }

        let mut candidates = Vec::new();

        if parent.is_dir() {
            for entry in fs::read_dir(&parent)? {
                let entry = entry?;

                if entry.file_type()?.is_dir() && entry.path().join("manifest.json").is_file() {
                    let manifest = Self::manifest(&entry.path())?;

                    if manifest["status"] == "completed" {
                        candidates.push(entry.path());
                    }
                }
            }
        }

        candidates.sort();

        if let Some(path) = candidates.pop() {
            return Ok(path);
        }

        let legacy = match (family, variant) {
            ("agent-collective", Some(name)) => root
                .join("artifacts")
                .join(format!("agent-collective-{name}")),

            ("agent-experimental", Some(name)) => {
                root.join("artifacts").join(format!("agent-{name}"))
            }

            _ => parent,
        };

        if legacy.join("massive.safetensors").is_file() {
            return Ok(legacy);
        }

        bail!("No trained AI available: {family}")
    }

    fn by_uuid(root: &Path, id: &str) -> Result<PathBuf> {
        let id = Uuid::parse_str(id)?.to_string();
        let base = root.join("artifacts");

        let mut stack = vec![(base, 0usize)];
        let mut result = None;

        while let Some((parent, depth)) = stack.pop() {
            if !parent.is_dir() {
                continue;
            }

            for entry in fs::read_dir(&parent)? {
                let entry = entry?;

                if !entry.file_type()?.is_dir() {
                    continue;
                }

                let path = entry.path();

                if entry.file_name().to_string_lossy() == id && path.join("manifest.json").is_file()
                {
                    if Self::manifest(&path)?["id"] == id {
                        if result.is_some() {
                            bail!("Duplicate UUID");
                        }

                        result = Some(path);
                    }
                } else if depth < 3 {
                    stack.push((path, depth + 1));
                }
            }
        }

        result.context("Unknown UUID")
    }

    fn validate(path: &Path) -> Result<()> {
        let manifest = Self::manifest(path)?;

        if manifest["status"] != "completed"
            || !path.join("massive-labels.json").is_file()
            || !path.join("source").is_dir()
            || !path.join("README.md").is_file()
        {
            bail!("Incomplete export");
        }

        Self::weights(path)?;
        Ok(())
    }

    fn copy_tree(from: &Path, to: &Path) -> Result<()> {
        fs::create_dir(to)?;

        for entry in fs::read_dir(from)? {
            let entry = entry?;
            let kind = entry.file_type()?;

            if kind.is_symlink() {
                bail!("Symlinks forbidden in export");
            }

            let source = entry.path();
            let destination = to.join(entry.file_name());

            if kind.is_dir() {
                Self::copy_tree(&source, &destination)?;
            } else if kind.is_file() {
                let mut input = fs::File::open(&source)?;

                let mut output = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&destination)?;

                std::io::copy(&mut input, &mut output)?;
            } else {
                bail!("Unsupported filesystem object");
            }
        }

        Ok(())
    }

    pub fn export(root: &Path, id: &str, destination: &Path) -> Result<()> {
        let source = Self::by_uuid(root, id)?;
        Self::validate(&source)?;

        if destination.exists() {
            bail!("Destination exists");
        }

        Self::copy_tree(&source, destination)?;
        Self::validate(destination)?;

        println!("Exported to {}", destination.display());
        Ok(())
    }

    pub fn import(root: &Path, source: &Path) -> Result<()> {
        Self::validate(source)?;

        let manifest = Self::manifest(source)?;

        let uuid = Uuid::parse_str(manifest["id"].as_str().context("Missing id")?)?;

        let family = manifest["family"].as_str().context("Missing family")?;

        let variant = manifest["variant"].as_str();
        let parent = Self::parent(root, family, variant)?;
        let destination = parent.join(uuid.to_string());

        if destination.exists() {
            bail!("Instance already exists");
        }

        fs::create_dir_all(&parent)?;
        Self::copy_tree(source, &destination)?;
        Self::validate(&destination)?;

        println!("Imported {}", destination.display());
        Ok(())
    }

    pub fn publish(root: &Path, args: &[String]) -> Result<()> {
        match args {
            [id, repo] if Uuid::parse_str(id).is_ok() => {
                if repo.starts_with('-')
                    || repo.split('/').count() != 2
                    || repo.split('/').any(str::is_empty)
                {
                    bail!("Expected namespace/model");
                }

                let source = Self::by_uuid(root, id)?;
                Self::validate(&source)?;

                let status = Command::new("hf")
                    .arg("upload")
                    .arg(repo)
                    .arg(&source)
                    .arg(".")
                    .status()?;

                if !status.success() {
                    bail!("Hugging Face upload failed: {status}");
                }

                Ok(())
            }

            _ => massive::run_massive(root, args),
        }
    }
}
