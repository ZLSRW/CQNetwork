from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.main.handler import train


SEQ_LEN = 41
NODE_FEATURE_DIM = 15
EMBED_DIM = 256
TARGET_REGION_LEN = 8
PROJECTION = np.random.default_rng(2024).normal(
    loc=0.0,
    scale=0.2,
    size=(NODE_FEATURE_DIM, EMBED_DIM),
).astype(np.float32)
MOTIF_PATTERNS = [re.compile(pattern) for pattern in (r"GGAC", r"GAC", r"AGAC")]
BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3}
COMPLEMENTARY_PAIRS = {
    ("A", "U"),
    ("U", "A"),
    ("C", "G"),
    ("G", "C"),
    ("G", "U"),
    ("U", "G"),
}
METRIC_NAMES = ["MCC", "AUC", "AUPR", "F1", "Acc", "Sen", "Spec", "Prec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GitHub-friendly CQNetwork runner for toy demos and fold-level experiments."
    )
    parser.add_argument("--sample_csv", type=str, default=None, help="Path to a toy CSV with sequence,label columns.")
    parser.add_argument("--train_npy", type=str, default=None, help="Path to a training fold .npy file.")
    parser.add_argument("--valid_npy", type=str, default=None, help="Path to a validation fold .npy file.")
    parser.add_argument("--output_dir", type=str, default="outputs/cqnetwork_demo", help="Directory for metrics and demo artifacts.")
    parser.add_argument("--device", type=str, default="auto", help="Device string, for example cpu, cuda:0, or auto.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size for the demo runner.")
    parser.add_argument("--lr", type=float, default=8e-3, help="Learning rate.")
    parser.add_argument("--valid_ratio", type=float, default=0.25, help="Validation split ratio for CSV demos.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def resolve_input_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None

    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (Path.cwd() / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sequence(sequence: str) -> str:
    sequence = (sequence or "").strip().upper().replace("T", "U")
    sequence = "".join(base if base in BASE_TO_INDEX else "U" for base in sequence)
    if len(sequence) < SEQ_LEN:
        sequence = sequence + ("U" * (SEQ_LEN - len(sequence)))
    return sequence[:SEQ_LEN]


def encode_node_features(sequence: str) -> np.ndarray:
    matrix = np.zeros((SEQ_LEN, NODE_FEATURE_DIM), dtype=np.float32)
    gc_total = 0
    au_total = 0

    for idx, base in enumerate(sequence):
        base_idx = BASE_TO_INDEX[base]
        position = idx / max(1, SEQ_LEN - 1)

        matrix[idx, base_idx] = 1.0
        matrix[idx, 4] = 1.0 if base in {"A", "G"} else 0.0
        matrix[idx, 5] = 1.0 if base in {"C", "U"} else 0.0
        matrix[idx, 6] = 1.0 if base in {"G", "C"} else 0.0
        matrix[idx, 7] = 1.0 if base in {"A", "U"} else 0.0
        matrix[idx, 8] = 1.0 if base in {"A", "C"} else 0.0
        matrix[idx, 9] = 1.0 if base in {"G", "U"} else 0.0
        matrix[idx, 10] = position
        matrix[idx, 11] = math.sin(2.0 * math.pi * position)
        matrix[idx, 12] = math.cos(2.0 * math.pi * position)

        if base in {"G", "C"}:
            gc_total += 1
        if base in {"A", "U"}:
            au_total += 1

        matrix[idx, 13] = gc_total / float(idx + 1)
        matrix[idx, 14] = au_total / float(idx + 1)

    return matrix


def build_primary_graph(sequence: str) -> np.ndarray:
    graph = np.eye(SEQ_LEN, dtype=np.float32)
    for idx in range(SEQ_LEN - 1):
        graph[idx, idx + 1] = 1.0
        graph[idx + 1, idx] = 1.0

    for i in range(SEQ_LEN):
        for j in range(i + 2, SEQ_LEN):
            if sequence[i] == sequence[j]:
                graph[i, j] = 0.35
                graph[j, i] = 0.35

    return graph


def build_loop_graph(sequence: str) -> np.ndarray:
    graph = np.eye(SEQ_LEN, dtype=np.float32)
    for i in range(SEQ_LEN):
        for j in range(i + 1, SEQ_LEN):
            if (sequence[i], sequence[j]) in COMPLEMENTARY_PAIRS:
                weight = 0.9 if abs(i - j) > 3 else 0.45
                graph[i, j] = weight
                graph[j, i] = weight
    return graph


def build_embedding(feature_matrix: np.ndarray) -> np.ndarray:
    position = np.arange(SEQ_LEN, dtype=np.float32)[:, None]
    dimension = np.arange(EMBED_DIM, dtype=np.float32)[None, :]
    positional_signal = np.sin((position + 1.0) / np.power(10000.0, (dimension % 32) / 32.0))
    embedding = feature_matrix @ PROJECTION
    return (embedding + 0.05 * positional_signal).astype(np.float32)


def infer_region_index(sequence: str) -> np.ndarray:
    for pattern in MOTIF_PATTERNS:
        match = pattern.search(sequence)
        if match:
            start = min(max(match.start() - 2, 0), SEQ_LEN - TARGET_REGION_LEN)
            return np.asarray([start, start + TARGET_REGION_LEN - 1], dtype=np.int64)

    start = max(0, (SEQ_LEN - TARGET_REGION_LEN) // 2)
    return np.asarray([start, start + TARGET_REGION_LEN - 1], dtype=np.int64)


def read_sequence_rows(csv_path: Path) -> list[tuple[str, int]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("The sample CSV must contain 'sequence' and 'label' columns.")

        rows = []
        for row in reader:
            sequence = normalize_sequence(row["sequence"])
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Labels must be 0 or 1, got {label!r}.")
            rows.append((sequence, label))

    if len(rows) < 4:
        raise ValueError("Provide at least four rows so the demo can create train and validation splits.")

    return rows


def split_rows(rows: list[tuple[str, int]], valid_ratio: float, seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    positives = [row for row in rows if row[1] == 1]
    negatives = [row for row in rows if row[1] == 0]
    if not positives or not negatives:
        raise ValueError("The sample CSV must contain both positive and negative examples.")

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    min_valid_per_class = 2 if len(positives) >= 4 and len(negatives) >= 4 else 1
    valid_pos = max(min_valid_per_class, min(len(positives) - 1, int(round(len(positives) * valid_ratio))))
    valid_neg = max(min_valid_per_class, min(len(negatives) - 1, int(round(len(negatives) * valid_ratio))))

    valid_rows = positives[:valid_pos] + negatives[:valid_neg]
    train_rows = positives[valid_pos:] + negatives[valid_neg:]
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    if len(train_rows) < 2 or len(valid_rows) < 2:
        raise ValueError("The split produced too few samples. Add more sample rows or lower --valid_ratio.")

    return train_rows, valid_rows


def rows_to_fold(rows: list[tuple[str, int]]) -> list[np.ndarray]:
    graphs = []
    loop_graphs = []
    flat_sequence_features = []
    embeddings = []
    region_indices = []

    for sequence, label in rows:
        feature_matrix = encode_node_features(sequence)
        graphs.append(build_primary_graph(sequence))
        loop_graphs.append(build_loop_graph(sequence))
        flat_sequence_features.append(
            np.concatenate([feature_matrix.reshape(-1), np.asarray([label], dtype=np.float32)], axis=0)
        )
        embeddings.append(build_embedding(feature_matrix))
        region_indices.append(infer_region_index(sequence))

    return [
        np.asarray(graphs, dtype=np.float32),
        np.asarray(loop_graphs, dtype=np.float32),
        np.asarray(flat_sequence_features, dtype=np.float32),
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(region_indices, dtype=np.int64),
    ]


def load_fold(path: Path) -> list:
    fold = np.load(path, allow_pickle=True).tolist()
    if not isinstance(fold, list) or len(fold) not in (4, 5):
        raise ValueError(f"{path} is not a supported CQNetwork fold file.")
    return fold


def build_training_args(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        train=True,
        evaluate=False,
        inverse=False,
        dataset="CQNetwork",
        epoch=args.epochs,
        lr=args.lr,
        device=device,
        validate_freq=1,
        batch_size=args.batch_size,
        batch_size1=max(1, args.batch_size - 1),
        norm_method="z_score",
        optimizer="RMSProp",
        early_stop=False,
        exponential_decay_step=20,
        decay_rate=0.1,
        dropout_rate=0.2,
        leakyrelu_rate=0.5,
        size=SEQ_LEN,
        num=1,
    )


def write_summary(output_dir: Path, metrics: list[float], config: dict) -> None:
    metric_payload = {}
    for name, value in zip(METRIC_NAMES, metrics):
        numeric_value = float(value)
        metric_payload[name] = numeric_value if math.isfinite(numeric_value) else None
    payload = {"config": config, "metrics": metric_payload}
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(payload, handle, indent=2)


def save_demo_fold(output_dir: Path, file_name: str, fold: list[np.ndarray]) -> None:
    payload = np.empty(len(fold), dtype=object)
    for idx, value in enumerate(fold):
        payload[idx] = value
    np.save(output_dir / file_name, payload, allow_pickle=True)


def main() -> None:
    args = parse_args()
    if bool(args.sample_csv) == bool(args.train_npy or args.valid_npy):
        raise ValueError("Use either --sample_csv or the pair --train_npy/--valid_npy.")
    if bool(args.train_npy) != bool(args.valid_npy):
        raise ValueError("Provide both --train_npy and --valid_npy together.")

    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_csv:
        csv_path = resolve_input_path(args.sample_csv)
        rows = read_sequence_rows(csv_path)
        train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
        train_data = rows_to_fold(train_rows)
        valid_data = rows_to_fold(valid_rows)
        save_demo_fold(output_dir, "sample_train_fold.npy", train_data)
        save_demo_fold(output_dir, "sample_valid_fold.npy", valid_data)
        species_name = "toy_sample"
    else:
        train_path = resolve_input_path(args.train_npy)
        valid_path = resolve_input_path(args.valid_npy)
        train_data = load_fold(train_path)
        valid_data = load_fold(valid_path)
        species_name = train_path.stem

    train_args = build_training_args(args, device)
    _, best_result = train(
        train_data=train_data,
        valid_data=valid_data,
        args=train_args,
        result_file=str(output_dir / "checkpoints"),
        fold=1,
        species=species_name,
    )

    config = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sample_csv": args.sample_csv,
        "train_npy": args.train_npy,
        "valid_npy": args.valid_npy,
    }
    write_summary(output_dir, best_result, config)

    print("CQNetwork run complete.")
    for metric_name, metric_value in zip(METRIC_NAMES, best_result):
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
