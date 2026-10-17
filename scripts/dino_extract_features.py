
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.eurosat import EuroSAT


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    metadata = {
        "model": "DINO ViT-S/16",
        "checkpoint": str(args.checkpoint_path.resolve()),
        "num_parameters": n_params,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "dataset": "EuroSAT RGB",
        "transform": (
            "Resize(256), CenterCrop(224), ToTensor(), "
            "ImageNet normalization"
        ),
        "feature_normalization": "None applied",
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