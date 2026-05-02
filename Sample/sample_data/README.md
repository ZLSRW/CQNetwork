# Sample Data

`cqnetwork_toy_sequences.csv` is a lightweight toy dataset for smoke testing the public CQNetwork repository.

Columns:

- `sequence`: RNA sequence string
- `label`: binary class label (`0` or `1`)

The toy runner converts each sequence into the CQNetwork tensor layout used by the main model:

- `41 x 15` node features
- `41 x 256` initial embeddings
- `41 x 41` primary graph
- `41 x 41` loop graph
- `2`-value target region index

To run the sample:

```bash
python run_cqnetwork.py \
  --sample_csv sample_data/cqnetwork_toy_sequences.csv \
  --device cpu \
  --epochs 1 \
  --batch_size 4 \
  --output_dir outputs/cqnetwork_toy_run
```

The runner will also export generated example folds as `.npy` files inside the chosen output directory.
