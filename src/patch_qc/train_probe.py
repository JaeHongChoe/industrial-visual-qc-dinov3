from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
import yaml

from .dataset import BalancedPatchDataset, collate_patch_batch
from .manifest import load_manifest_records
from .model import load_huggingface_probe, load_timm_probe
from .train import evaluate_binary, save_probe_checkpoint, train_one_epoch


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _loader(dataset: BalancedPatchDataset, config: dict[str, object], shuffle: bool) -> DataLoader:
    workers = int(config.get("workers", 8))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_patch_batch,
    )


def run(config_path: str | Path, manifest_path: str | Path, output_dir: str | Path) -> int:
    config = _load_config(config_path)
    model_config = dict(config["model"])
    data_config = dict(config["data"])
    train_config = dict(config["train"])
    if not torch.cuda.is_available():
        raise RuntimeError("probe training requires a Slurm-allocated CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one Slurm-visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.manual_seed(int(train_config.get("seed", 17)))
    torch.cuda.manual_seed_all(int(train_config.get("seed", 17)))
    torch.set_float32_matmul_precision("high")

    train_records = load_manifest_records(manifest_path, split="train")
    holdout_records = load_manifest_records(manifest_path, split="holdout")
    if not train_records or not holdout_records:
        raise RuntimeError("manifest must contain non-empty train and holdout splits")
    train_dataset = BalancedPatchDataset(
        train_records,
        patch_size=int(data_config.get("patch_size", 256)),
        samples_per_epoch=int(train_config.get("samples_per_epoch", 20_000)),
        positive_probability=0.5,
        seed=int(train_config.get("seed", 17)),
        source_reuse=int(data_config.get("patches_per_source", 8)),
    )
    evaluation_dataset = BalancedPatchDataset(
        holdout_records,
        patch_size=int(data_config.get("patch_size", 256)),
        samples_per_epoch=int(train_config.get("evaluation_samples", 2_000)),
        positive_probability=0.5,
        seed=int(train_config.get("seed", 17)) + 10_000,
        source_reuse=int(data_config.get("patches_per_source", 8)),
    )
    train_loader = _loader(train_dataset, train_config, shuffle=False)
    evaluation_loader = _loader(evaluation_dataset, train_config, shuffle=False)

    provider = str(model_config.get("provider", "timm"))
    if provider == "timm":
        model = load_timm_probe(
            str(model_config["backbone_id"]),
            local_files_only=bool(model_config.get("local_files_only", True)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(device)
    elif provider == "huggingface":
        model = load_huggingface_probe(
            str(model_config["backbone_id"]),
            token=os.environ.get("HF_TOKEN"),
            local_files_only=bool(model_config.get("local_files_only", True)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).to(device)
    else:
        raise ValueError(f"unsupported model provider: {provider}")
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(train_config.get("learning_rate", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    epochs = int(train_config.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        training = train_one_epoch(model, train_loader, optimizer, device=device, use_amp=True)
        evaluation = evaluate_binary(model, evaluation_loader, device=device)
        row = {"epoch": epoch, "train": training, "synthetic_holdout": evaluation}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        save_probe_checkpoint(
            model,
            destination / f"head_epoch_{epoch:03d}.pt",
            backbone_id=str(model_config["backbone_id"]),
            epoch=epoch,
            metrics={
                "train_loss": float(training["loss"]),
                "train_accuracy": float(training["accuracy"]),
                "holdout_loss": float(evaluation["loss"]),
                "holdout_accuracy": float(evaluation["accuracy"]),
                "examples": int(evaluation["examples"]),
            },
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patch-qc-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    return run(args.config, args.manifest, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
