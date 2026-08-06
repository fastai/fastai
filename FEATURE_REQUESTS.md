# Feature Requests

This file tracks community-requested features for fastai. If you would like to suggest a new feature, please add a one-line description below under the appropriate category and submit a PR.

For discussions about features before implementation, please use the [fastai forum](https://forums.fast.ai/).

## Data

- Add built-in support for streaming/iterable datasets that do not fit in memory (e.g. webdataset or HuggingFace IterableDataset integration)

## Training

- Support learning rate finder (`lr_find`) with multiple losses displayed on the same plot for multi-task models

## Vision

- Add native support for ONNX model export with dynamic batch-size axes directly from `Learner`

## Text

- Provide a high-level API for parameter-efficient fine-tuning (LoRA/QLoRA adapters) on large language models

## Tabular

- Allow incremental/online learning for tabular models so new data can be incorporated without full retraining

## Callbacks

- Add a built-in EarlyStopping callback that supports monitoring multiple metrics with configurable logic (any/all)

## Deployment & Export

- Provide a `Learner.to_api()` convenience method that generates a minimal FastAPI/Flask prediction endpoint from a trained model

## Documentation & Tooling

- Add a CLI command (`fastai_check_env`) that validates GPU drivers, CUDA version, and dependency compatibility in one step
