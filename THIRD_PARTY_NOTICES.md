# Third-Party Notices

This repository integrates third-party software and pretrained model weights. The Apache-2.0
license in [LICENSE](LICENSE) applies only to original workbench source and documentation unless a
file states otherwise.

## Google Research TabFM source

- Project: [Google Research TabFM](https://github.com/google-research/tabfm)
- Pinned revision: `cb6ba46b7ebc9a6581a81827e14e9c246202afb9`
- Upstream source license: [Apache License 2.0](https://github.com/google-research/tabfm/blob/cb6ba46b7ebc9a6581a81827e14e9c246202afb9/LICENSE)
- Integration: installed as a direct uv dependency and imported at runtime

The workbench does not vendor or modify upstream TabFM source files.

## TabFM v1.0.0 PyTorch weights

- Model repository: [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- Weight license: [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)
- Tasks: classification and regression checkpoints

The model-weight license allows only specified non-commercial, non-production purposes and
restricts commercial use, production use, derivatives, and distribution. Downloading or using the
weights constitutes acceptance of those upstream terms. This repository does not redistribute the
weights, grant a commercial license, or modify their license.

## Other dependencies

Python dependencies and resolved versions are declared in `pyproject.toml` and `uv.lock`. Each
package remains subject to its own license. Inspect installed metadata when redistributing this
workbench or a bundled environment.

No Google trademark license or endorsement is granted. TabFM is not an officially supported Google
product.
