# CQNetwork

CQNetwork is a consensus-knowledge-guided graph neural network for RNA site classification. The repository couples sequence-derived node features, motif-aware graph construction, consensus graph reasoning, and graph wavelet propagation in a single end-to-end model.

This bundle includes:

- a GitHub-friendly runner in `run_cqnetwork.py`
- dependency definitions in `requirements.txt`
- a lightweight sample dataset in `sample_data/`
- the full CQNetwork model implementation in `models/main/`
- the original experiment entry script in `main.py`
- the fold loader in `data_loader/SiteBinding_dataloader1.py`

## Repository Layout

```text
github_release/
├── data_loader/
│   ├── __init__.py
│   └── SiteBinding_dataloader1.py
├── models/
│   ├── __init__.py
│   └── main/
│       ├── __init__.py
│       ├── ConsensusComponents.py
│       ├── ConsensusNetwork.py
│       ├── Graph_Wavelet.py
│       ├── SemanticNet.py
│       ├── Utils.py
│       ├── configure.py
│       └── handler.py
├── main.py
├── README.md
├── requirements.txt
├── run_cqnetwork.py
└── sample_data/
    ├── README.md
    └── cqnetwork_toy_sequences.csv
```

## Core Modules

- `models/main/ConsensusComponents.py`: key-region filtering, motif graph construction, consensus-node preparation, and contrastive representation learning.
- `models/main/ConsensusNetwork.py`: the top-level CQNetwork architecture and graph fusion logic.
- `models/main/Graph_Wavelet.py`: graph wavelet neural network layers.
- `models/main/SemanticNet.py`: sequence semantic feature extractor.
- `models/main/handler.py`: training and validation utilities.
- `data_loader/SiteBinding_dataloader1.py`: fold loader for CQNetwork training data.

## Requirements

Install the runtime dependencies with:

```bash
cd github_release
pip install -r requirements.txt
```

The main dependencies are:

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- pandas
- scikit-learn

## Quick Start

The fastest way to verify the repository is to run the included toy example:

```bash
cd github_release
python run_cqnetwork.py \
  --sample_csv sample_data/cqnetwork_toy_sequences.csv \
  --device cpu \
  --epochs 1 \
  --batch_size 4 \
  --output_dir outputs/cqnetwork_toy_run
```

This command will:

1. read the CSV file in `sample_data/`
2. construct toy CQNetwork inputs with the expected tensor shapes
3. run a short training/validation cycle through the real CQNetwork model
4. write `metrics.json` and generated fold files into `outputs/cqnetwork_toy_run/`

## Running on Real Fold Files

`run_cqnetwork.py` also supports direct fold-level training with `.npy` files when you place your prepared folds under the repository tree:

```bash
cd github_release
python run_cqnetwork.py \
  --train_npy Pre-Encoding/data/rat_liver/Train_Test/all/TrainData1.npy \
  --valid_npy Pre-Encoding/data/rat_liver/Train_Test/all/TestData1.npy \
  --device cuda:0 \
  --epochs 100 \
  --batch_size 128 \
  --output_dir outputs/rat_liver_fold1
```

Supported fold layouts:

- 4-part folds: `[graph, loop_graph, flattened_sequence_features_plus_label, embedding]`
- 5-part folds: `[graph, loop_graph, flattened_sequence_features_plus_label, embedding, region_index]`

When `region_index` is not present, the loader reconstructs it automatically from the flattened sequence features.

## Input Format

### Sample CSV

The included toy CSV uses two columns:

- `sequence`: RNA sequence string
- `label`: binary label (`0` or `1`)

The demo runner normalizes each sequence to 41 nt by converting `T` to `U`, removing unsupported characters, and padding or truncating as needed.

### Fold Tensors

CQNetwork expects the following internal shapes:

- graph: `[N, 41, 41]`
- loop graph: `[N, 41, 41]`
- flattened sequence features + label: `[N, 616]`
- embedding: `[N, 41, 256]`
- region index: `[N, 2]` (optional in stored folds, reconstructed if missing)

The 616-dimensional vector corresponds to:

- `41 x 15 = 615` sequence/node features
- `1` binary label

## Original Experiment Script

The repository keeps `main.py` as the original experiment-oriented entry script. It is useful for legacy workflows tied to the original directory layout under `Pre-Encoding/data/...`.

For public usage, `run_cqnetwork.py` is the recommended entrypoint because it:

- resolves CPU/GPU devices automatically
- works with both 4-part and 5-part fold files
- supports a toy sample for smoke testing
- saves a compact metrics summary

## Notes

- The sample dataset is intentionally tiny and is only meant to validate the code path.
- The toy runner generates surrogate 256-dimensional initial embeddings from handcrafted sequence features so that the full CQNetwork stack can be executed without shipping the original large data assets.
- For research-scale experiments, use the original fold files and adjust epochs, batch size, and output directories as needed.
