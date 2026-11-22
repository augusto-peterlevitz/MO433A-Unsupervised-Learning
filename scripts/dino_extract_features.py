
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.eurosat import EuroSAT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEACHER_SOURCE = "https://github.com/facebookresearch/dino"
TEACHER_PAPER = "https://arxiv.org/abs/2104.14294"


def parse_args():
    """Parse dataset, checkpoint, output, and data-loading configuration."""
    parser = argparse.ArgumentParser(
        description="Extract DINO ViT-S/16 features for the EuroSAT dataset."
    )
    parser.add_argument(
        "--eurosat-root",
        type=Path,
        default=Path("datasets/EuroSAT_RGB"),
        help="Path to the EuroSAT dataset directory (default: datasets/EuroSAT_RGB)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("dino_deitsmall16_pretrain.pth"),
        help="Path to the pretrained DINO checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/features/dino_vits16"),
        help="Directory where extracted features and metadata are saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images per batch (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data-loading worker processes (default: 4).",
    )
    return parser.parse_args()


# ---- Deterministic teacher preprocessing ----
teacher_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def main():
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DINO checkpoint not found: {args.checkpoint_path}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset in its existing, deterministic order.
    dataset = EuroSAT(
        root=args.eurosat_root,
        transform=teacher_transform,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # 2. Build the frozen DINO ViT-S/16 teacher.
    teacher = torch.hub.load(
        "facebookresearch/dino:main",
        "dino_vits16",
        pretrained=False,
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_sha256 = hashlib.sha256()
    with args.checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_sha256.update(chunk)

    teacher.load_state_dict(checkpoint, strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(DEVICE)

    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"Device: {DEVICE}")
    print(f"Parameters: {n_params:,}")

    # 3. Extract features without gradients.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Preserve dataset order!
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )

    all_features = []
    all_labels = []

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)

            features = teacher(images)

            if features.ndim != 2 or features.shape[1] != 384:
                raise RuntimeError(
                    f"Unexpected feature shape: {tuple(features.shape)}"
                )

            if not torch.isfinite(features).all():
                raise RuntimeError("Non-finite DINO features detected.")

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if batch_idx % 20 == 0:
                print(
                    f"Processed "
                    f"{min((batch_idx + 1) * args.batch_size, len(dataset))}"
                    f"/{len(dataset)} images"
                )

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    # 4. Save features, labels, and provenance metadata.
    np.save(args.output_dir / "features.npy", features)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "indices.npy", np.arange(len(dataset)))

    metadata = {
        "model": "DINO ViT-S/16",
        "approach": "teacher_encoding",
        "checkpoint": str(args.checkpoint_path.resolve()),
        "teacher_provenance": {
            "checkpoint_name": args.checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256.hexdigest(),
            "source_repository": TEACHER_SOURCE,
            "source_model": "dino_vits16 (DeiT-S/16 / ViT-S/16)",
            "paper": TEACHER_PAPER,
            "pretraining_dataset": (
                "ImageNet-1k training images; self-supervised, labels not used"
            ),
            "pretraining_compute_references": [
                {
                    "run": "DINO ViT-S/16, 100 epochs",
                    "reported_wall_time_days": 1.75,
                    "reported_gpu_count": 8,
                    "derived_gpu_days": 336.0,
                },
                {
                    "run": "Boosted DINO ViT-S/16, 300 epochs",
                    "reported_wall_time_days": 2.6,
                    "reported_gpu_count": 16,
                    "derived_gpu_days": 998.4,
                },
            ],
            "compute_attribution": (
                "The cited repository reports these reference runs. The exact "
                "training run corresponding to this checkpoint file has not "
                "been verified; do not treat either compute reference as its "
                "confirmed cost."
            ),
        },
        "num_parameters": n_params,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "dataset": "EuroSAT RGB",
        "transform": (
            "Resize(256), CenterCrop(224), ToTensor(), "
            "ImageNet normalization"
        ),
        "feature_normalization": "None applied",
        "feature_pooling": "CLS token; output of dino_vits16 forward",
        "shuffle": False,
    }

    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # 5. Report final sanity checks.
    print("\n--- Feature cache verification ---")
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("All features finite:", np.isfinite(features).all())
    print("Feature mean:", float(features.mean()))
    print("Feature std:", float(features.std()))
    print("Unique labels:", np.unique(labels))
    print("Saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()