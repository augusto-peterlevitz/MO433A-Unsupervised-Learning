
import argparse
import json
import pickle
import random
import time
from contextlib import closing
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from lightly.loss import VICRegLoss
from lightly.transforms.vicreg_transform import VICRegTransform
from tqdm import tqdm

from data.dataloaders import UnlabeledDataset
from models.student_cnn import build_student
from models.vicreg import VICRegModel
from data.eurosat import EuroSAT


@dataclass
class TrainingConfig:
    experiment_name: str = "vicreg_training"
    dataset_root: Path = Path("datasets/EuroSAT_RGB")
    subsets_path: Path = Path("datasets/eurosat_train_subsets.pkl")
    student_cnn: str = "large"
    approach: str = "vicreg"
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    output_dir: Path = Path("outputs/checkpoints/vicreg")

    def validate(self):
        if (
            not self.experiment_name.strip()
            or "/" in self.experiment_name
            or "\\" in self.experiment_name
        ):
            raise ValueError(
                "experiment_name must be non-empty and cannot contain path separators"
            )
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError(
                "epochs and batch_size must be positive; "
                "num_workers must be nonnegative"
            )
        if self.student_cnn not in {"small", "large"}:
            raise ValueError("student_cnn must be 'small' or 'large'")
        if self.approach not in {
            "distillation",
            "vicreg",
            "random_baseline",
            "teacher_encoding",
        }:
            raise ValueError(
                "approach must be one of: distillation, vicreg, "
                "random_baseline, teacher_encoding"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")


def load_config(path):
    with open(path, encoding="utf-8") as config_file:
        values = json.load(config_file)
    if not isinstance(values, dict):
        raise ValueError("Training configuration must be a JSON object")

    if "row" in values and "approach" not in values:
        legacy_row = values.pop("row")
        legacy_approaches = {"B": "vicreg", "C": "distillation"}
        try:
            values["approach"] = legacy_approaches[legacy_row]
        except KeyError as error:
            raise ValueError(f"Unsupported legacy row value: {legacy_row!r}") from error

    config_fields = {field.name: field for field in fields(TrainingConfig)}
    unknown_fields = set(values) - set(config_fields)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown training configuration fields: {names}")

    path_fields = {"dataset_root", "subsets_path", "output_dir"}
    string_fields = {"experiment_name", "approach", "student_cnn"}
    integer_fields = {"batch_size", "epochs", "num_workers", "seed"}
    float_fields = {"learning_rate", "weight_decay"}
    for name, value in values.items():
        if name in path_fields:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string path")
            values[name] = Path(value)
        elif name in string_fields and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
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
        description="Train row-B VICReg students using a JSON configuration file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="Path to the base training configuration (default: configs/vicreg.json).",
    )
    parser.add_argument(
        "--student-cnn",
        choices=("small", "large"),
        help="Train every requested seed for this StudentCNN architecture.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Random seeds; use at least 3 for small and 2 for large.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Root directory for size- and seed-specific checkpoints.",
    )
    return parser.parse_args()


def train(config):
    random.seed(config.seed)
    if config.approach != "vicreg":
        raise ValueError("training.train_vicreg requires approach='vicreg'.")
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

    dataset = UnlabeledDataset(
        EuroSAT(
            root=config.dataset_root,
            transform=VICRegTransform(input_size=64),
        )
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

    backbone = build_student(config.student_cnn)
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
    config_path = config.output_dir / "vicreg.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(
            {
                name: str(value) if isinstance(value, Path) else value
                for name, value in asdict(config).items()
            },
            config_file,
            indent=2,
        )
        config_file.write("\n")

    started_at = datetime.now().astimezone()
    run_timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{run_timestamp}"
    tensorboard_dir = config.output_dir / "tensorboard" / run_name
    with closing(SummaryWriter(log_dir=str(tensorboard_dir))) as summary_writer:
        summary_writer.add_text("experiment/name", config.experiment_name, 0)
        summary_writer.add_text("experiment/started_at", started_at.isoformat(), 0)
        global_step = 0
        epoch_progress = tqdm(range(config.epochs), desc="Training", unit="epoch")
        for epoch in epoch_progress:
            model.train()
            total_loss = 0.0
            num_batches = 0
            start_time = time.time()

            batch_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.epochs}",
                unit="batch",
                leave=False,
            )
            for views in batch_progress:
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

                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1
                global_step += 1
                summary_writer.add_scalar(
                    "train/batch_loss", batch_loss, global_step
                )
                batch_progress.set_postfix(loss=f"{batch_loss:.4f}")

            if num_batches == 0:
                raise RuntimeError(
                    "Training DataLoader produced no batches; "
                    "check the subset size and batch_size."
                )

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
            summary_writer.add_scalar("train/epoch_loss", average_loss, epoch + 1)
            summary_writer.add_scalar("train/learning_rate", current_lr, epoch + 1)
            summary_writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch + 1)
            summary_writer.flush()
            epoch_progress.set_postfix(
                loss=f"{average_loss:.4f}", lr=f"{current_lr:.2e}"
            )

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": average_loss,
                "student_cnn": config.student_cnn,
            }
            torch.save(
                checkpoint,
                config.output_dir / "last_checkpoint.pth",
            )

    backbone_path = config.output_dir / "student_backbone.pth"
    torch.save(model.backbone.state_dict(), backbone_path)
    print("VICReg training finished.")
    print(f"Configuration saved to: {config_path}")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"Backbone saved to: {backbone_path}")


def main():
    args = parse_args()
    base_config = load_config(args.config)

    if (args.student_cnn is None) != (args.seeds is None):
        raise ValueError("--student-cnn and --seeds must be provided together.")

    if args.student_cnn is None:
        train(base_config)
        return

    minimum_seeds = 3 if args.student_cnn == "small" else 2
    if len(args.seeds) < minimum_seeds:
        raise ValueError(
            f"The {args.student_cnn} student requires at least {minimum_seeds} seeds."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds cannot contain duplicates.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain nonnegative integers.")

    output_root = args.output_dir or Path("outputs/checkpoints/vicreg")
    size_name = "small_cnn" if args.student_cnn == "small" else "larger_cnn"
    for seed in args.seeds:
        config = replace(
            base_config,
            experiment_name=f"vicreg_{size_name}",
            student_cnn=args.student_cnn,
            approach="vicreg",
            seed=seed,
            output_dir=output_root / size_name / f"seed{seed}",
        )
        config.validate()
        print(
            f"\nStarting row B: student_cnn={args.student_cnn}, seed={seed}"
        )
        train(config)


if __name__ == "__main__":
    main()
