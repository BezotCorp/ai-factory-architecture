# AI Factory Architecture

**Experimental research into alternative artificial intelligence architectures.**

AI Factory Architecture is an experimental project exploring whether unconventional computational structures can learn to perform real-world tasks efficiently.

The goal is not to reproduce existing architectures such as Transformers or Mamba, but to investigate alternative ways of organizing computation, memory, information flow, and learning.

## Objectives

- Design and experiment with unconventional AI architectures.
- Train models from scratch using existing datasets.
- Evaluate their ability to learn practical tasks, including natural language processing.
- Explore computational efficiency, memory consumption, and generalization.
- Compare experimental approaches against measurable baselines.

## Research direction

The project begins without a predefined architecture.

Possible research directions include adaptive circuits, heterogeneous computational operations, associative memory, dynamic connectivity, and alternative learning mechanisms.

These are possibilities to investigate, not established design decisions.

## Experimental approach

Experiments will use reproducible datasets with defined inputs and expected outputs.

Results should be evaluated through task performance, generalization, computational cost, and resource consumption.

No specific programming language, framework, architecture, or learning algorithm has been selected.

## Status

**Early research — architecture not yet defined.**

This repository will document the project's design decisions, experiments, results, and limitations as research progresses.
## Publishing the MASSIVE model

The laboratory can prepare and publish its trained associative-circuits model
without retraining. It uses `artifacts/associative-circuits/massive.safetensors`
and `massive-labels.json`; keep these files together.

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
and authenticate with `hf auth login` using an account with write access. First,
prepare the files in a **new** directory and review the generated model card:

```sh
cargo run --no-default-features -p laboratory -- publish --prepare-only /tmp/massive-model-review
```

Edit the generated card as needed, then explicitly publish the reviewed directory:

```sh
cargo run --no-default-features -p laboratory -- publish YOUR_NAMESPACE/YOUR_MODEL --from /tmp/massive-model-review
```

The `hf upload` command may create the model repository if it does not exist.
Check its visibility and your account permissions before publishing. Omit `--from` to publish a freshly prepared temporary copy of the current model.
Neither mode modifies the local weights or labels.
The generated card describes the current architecture; add verified evaluation
results, usage details, licensing, and attribution as appropriate before sharing.
