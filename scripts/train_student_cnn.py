import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from data.eurosat import EuroSAT
from models.student_cnn import StudentCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedStudent(nn.Module):
    """StudentCNN encoder with a linear image-classification head."""

    def __init__(self, num_classes, width_mult):
        super().__init__()
        self.encoder = StudentCNN(width_mult=width_mult)
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)

    def forward(self, images):
        return self.classifier(self.encoder(images))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the student CNN to classify EuroSAT images."
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("datasets/EuroSAT_RGB"),
        help="Path to the EuroSAT image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/student_cnn"),
        help="Directory for epoch metrics and the best model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_split(labels, validation_fraction, seed):
    """Return deterministic train and validation indices preserving class ratios."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        if len(class_indices) < 2:
            raise ValueError(
                "Each class must have at least two images for a "
                "stratified train/validation split."
            )

        rng.shuffle(class_indices)
        validation_count = min(
            len(class_indices) - 1,
            max(1, int(round(len(class_indices) * validation_fraction))),
        )
        validation_indices.extend(class_indices[:validation_count])
        train_indices.extend(class_indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    dataset = EuroSAT(root=args.eurosat_root, transform=transform)
    if not dataset.samples:
        raise ValueError(f"No EuroSAT images found under {args.eurosat_root}")

    labels = np.asarray([label for _, label in dataset.samples])
    train_indices, validation_indices = stratified_split(
        labels, args.validation_fraction, args.seed
    )
    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
    )

    model = SupervisedStudent(
        num_classes=len(dataset.classes),
        width_mult=args.width_mult,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    checkpoint_path = args.output_dir / "best_model.pt"
    best_validation_accuracy = -1.0

    print(f"Device: {DEVICE}")
    print(
        f"Samples: {len(dataset)} "
        f"(train={len(train_dataset)}, validation={len(validation_dataset)})"
    )
    print(f"Classes: {dataset.classes}")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {checkpoint_path}")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion
            )
            learning_rate = optimizer.param_groups[0]["lr"]

            writer.writerow({
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            })
            metrics_file.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} "
                f"val_loss={validation_loss:.4f} "
                f"val_acc={validation_accuracy:.4f}"
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": dataset.class_to_idx,
                        "width_mult": args.width_mult,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                    },
                    checkpoint_path,
                )

    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")


if __name__ == "__main__":
    main()
