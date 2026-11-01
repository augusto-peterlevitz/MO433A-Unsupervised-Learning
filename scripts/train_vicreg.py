
import argparse
import csv
import json
import pickle
import random
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lightly.loss import VICRegLoss
from lightly.transforms.vicreg_transform import VICRegTransform

from models.student_cnn import StudentCNN
from models.vicreg import VICRegModel
from data.eurosat import EuroSAT


@dataclass
class TrainingConfig:
    dataset_root: Path = Path("datasets/EuroSAT_RGB")
    subsets_path: Path = Path("datasets/eurosat_train_subsets.pkl")
    width_mult: float = 1.0
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    output_dir: Path = Path("checkpoints/vicreg_width_1.0")

    def validate(self):
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError(
                "epochs and batch_size must be positive; "
                "num_workers must be nonnegative"
            )
        if self.width_mult <= 0 or self.learning_rate <= 0:
            raise ValueError("width_mult and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")


def load_config(path):
    with open(path, encoding="utf-8") as config_file:
        values = json.load(config_file)
    if not isinstance(values, dict):
        raise ValueError("Training configuration must be a JSON object")

    config_fields = {field.name: field for field in fields(TrainingConfig)}
    unknown_fields = set(values) - set(config_fields)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown training configuration fields: {names}")

    path_fields = {"dataset_root", "subsets_path", "output_dir"}
    integer_fields = {"batch_size", "epochs", "num_workers", "seed"}
    float_fields = {"width_mult", "learning_rate", "weight_decay"}
    for name, value in values.items():
        if name in path_fields:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string path")
            values[name] = Path(value)
        elif name in integer_fields and type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        elif name in float_fields:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            values[name] = float(value)

    config = TrainingConfig(**values)
    config.validate()
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a VICReg model using a JSON configuration file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="Path to the training configuration (default: configs/vicreg.json).",
    )
    return parser.parse_args()


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    with open(config.subsets_path, "rb") as subset_file:
        train_subsets = pickle.load(subset_file)

    # Use all training images, not the test images.
    train_indices = train_subsets[(1.0, 0)]
    print(f"Number of training images: {len(train_indices)}")

    dataset = EuroSAT(
        root=config.dataset_root,
        transform=VICRegTransform(input_size=64),
    )
    train_dataset = Subset(dataset, list(map(int, train_indices)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )

    backbone = StudentCNN(width_mult=config.width_mult)
    model = VICRegModel(backbone).to(device)

    print(f"Student feature dimension: {backbone.feature_dim}")
    print(
        f"Number of model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = VICRegLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "history.csv"

    with open(history_path, "w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "loss", "learning_rate", "seconds"])

        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            start_time = time.time()

            for views, _ in train_loader:
                # The dataset returns two independently augmented views.
                x0, x1 = views
                x0 = x0.to(device, non_blocking=True)
                x1 = x1.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Mixed precision for the model's forward passes.
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    z0 = model(x0)
                    z1 = model(x1)

                # Calculate the VICReg loss in float32 for numerical stability.
                loss = criterion(z0.float(), z1.float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            scheduler.step()
            epoch_seconds = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1:02d}/{config.epochs}] "
                f"Loss: {average_loss:.4f} "
                f"LR: {current_lr:.6f} "
                f"Time: {epoch_seconds:.1f}s"
            )
            writer.writerow([
                epoch + 1,
                average_loss,
                current_lr,
                epoch_seconds,
            ])
            history_file.flush()

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": average_loss,
                "width_mult": config.width_mult,
            }
            torch.save(
                checkpoint,
                config.output_dir / "last_checkpoint.pth",
            )

    backbone_path = config.output_dir / "student_backbone.pth"
    torch.save(model.backbone.state_dict(), backbone_path)
    print("VICReg training finished.")
    print(f"Backbone saved to: {backbone_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
