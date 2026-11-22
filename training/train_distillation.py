import argparse
import csv
import json
import pickle
import random
import time
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.transforms.vicreg_transform import VICRegTransform
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.eurosat import EuroSAT
from data.dataloaders import UnlabeledDataset
from models.student_cnn import build_student
from training.train_vicreg import load_config

TEACHER_DIM = 384
TEACHER_SIZE = 224


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a StudentCNN by distilling a frozen DINO ViT-S/16."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vicreg.json"),
        help="Base VICReg config; its training budget and data paths are reused.",
    )
    parser.add_argument(
        "--student-cnn",
        choices=("small", "large"),
        required=True,
        help="Student architecture to train.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds; use at least 3 for small and 2 for large.",
    )
    parser.add_argument(
        "--projector-type",
        choices=("mlp", "linear"),
        default="mlp",
        help=(
            "Student-to-teacher projector. Use linear for the required projector "
            "ablation; mlp is the main-comparison default."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("dino_deitsmall16_pretrain.pth"),
        help="Local pretrained DINO ViT-S/16 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoints/distillation"),
        help="Root for size- and seed-specific runs.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINO teacher checkpoint not found: {checkpoint_path}")

    teacher = torch.hub.load(
        "facebookresearch/dino:main",
        "dino_vits16",
        pretrained=False,
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.to(device)

    parameter_count = sum(parameter.numel() for parameter in teacher.parameters())
    print(f"Teacher: DINO ViT-S/16 ({parameter_count:,} parameters)")
    return teacher


def save_config(config, output_path):
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2)
        file.write("\n")


def build_projector(student_dim, projector_type):
    if projector_type == "linear":
        return nn.Linear(student_dim, TEACHER_DIM)
    if projector_type == "mlp":
        return nn.Sequential(
            nn.Linear(student_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, TEACHER_DIM),
        )
    raise ValueError(f"Unsupported projector type: {projector_type}")


def train_one(config, teacher, teacher_checkpoint, projector_type):
    config.validate()
    if config.approach != "distillation":
        raise ValueError("train_distillation requires approach='distillation'.")
    if projector_type == "mlp" and config.batch_size < 2:
        raise ValueError("Distillation batch_size must be at least 2 for projector BatchNorm.")
    set_seed(config.seed)
    device = next(teacher.parameters()).device
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True

    with config.subsets_path.open("rb") as file:
        train_subsets = pickle.load(file)
    train_indices = train_subsets[(1.0, 0)]

    dataset = UnlabeledDataset(
        EuroSAT(
            root=config.dataset_root,
            transform=VICRegTransform(input_size=64),
        )
    )
    train_dataset = Subset(dataset, list(map(int, train_indices)))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
        generator=generator,
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "The unlabeled training split has fewer samples than one full batch."
        )

    student = build_student(config.student_cnn).to(device)
    projector = build_projector(student.feature_dim, projector_type).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config.output_dir / "vicreg.json")
    metrics_path = config.output_dir / "metrics.csv"
    run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"distillation_{projector_type}_{config.student_cnn}_"
        f"seed{config.seed}_{run_timestamp}"
    )
    tensorboard_dir = config.output_dir / "tensorboard" / run_name
    teacher_timing_events = []
    teacher_forward_seconds = 0.0
    global_step = 0
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    print(
        f"Distillation: projector={projector_type}, student={config.student_cnn}, "
        f"seed={config.seed}, "
        f"feature_dim={student.feature_dim}, parameters={parameter_count:,}, "
        f"images={len(train_dataset)}, batches/epoch={len(train_loader)}"
    )

    with closing(SummaryWriter(log_dir=str(tensorboard_dir))) as summary_writer, \
            metrics_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
        summary_writer.add_text("experiment/approach", config.approach, 0)
        summary_writer.add_text("experiment/student_cnn", config.student_cnn, 0)
        summary_writer.add_text("experiment/seed", str(config.seed), 0)
        summary_writer.add_text("experiment/projector_type", projector_type, 0)
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "loss", "learning_rate", "epoch_seconds"),
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            student.train()
            projector.train()
            epoch_started = time.perf_counter()
            total_loss = 0.0

            for views in tqdm(
                train_loader,
                desc=f"{config.student_cnn} seed {config.seed}, epoch {epoch}/{config.epochs}",
                leave=False,
            ):
                # Row B uses this same VICReg augmentation. Row C uses one
                # augmented view for both networks, then upsamples it for DINO.
                student_images = views[0].to(device, non_blocking=True)
                teacher_images = F.interpolate(
                    student_images.float(),
                    size=(TEACHER_SIZE, TEACHER_SIZE),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        teacher_features = teacher(teacher_images)
                    end_event.record()
                    teacher_timing_events.append((start_event, end_event))
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        student_features = student(student_images)
                        projected_features = projector(student_features)
                else:
                    teacher_started = time.perf_counter()
                    with torch.no_grad():
                        teacher_features = teacher(teacher_images)
                    teacher_forward_seconds += time.perf_counter() - teacher_started
                    student_features = student(student_images)
                    projected_features = projector(student_features)

                if (
                    teacher_features.ndim != 2
                    or teacher_features.shape[1] != TEACHER_DIM
                ):
                    raise RuntimeError(
                        f"Expected teacher features [batch, {TEACHER_DIM}], got "
                        f"{tuple(teacher_features.shape)}."
                    )

                loss = 1.0 - F.cosine_similarity(
                    projected_features.float(),
                    teacher_features.float(),
                    dim=-1,
                ).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite distillation loss at epoch {epoch}."
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_loss = float(loss.detach())
                total_loss += batch_loss
                global_step += 1
                summary_writer.add_scalar(
                    "train/batch_loss",
                    batch_loss,
                    global_step,
                )

            average_loss = total_loss / len(train_loader)
            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_started
            learning_rate = optimizer.param_groups[0]["lr"]
            writer.writerow({
                "epoch": epoch,
                "loss": average_loss,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
            })
            file.flush()
            summary_writer.add_scalar("train/epoch_loss", average_loss, epoch)
            summary_writer.add_scalar("train/learning_rate", learning_rate, epoch)
            summary_writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
            summary_writer.flush()
            print(
                f"Epoch {epoch:03d}/{config.epochs} "
                f"loss={average_loss:.5f} lr={learning_rate:.6g} "
                f"time={epoch_seconds:.1f}s"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "projector_state_dict": projector.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": average_loss,
                    "student_cnn": config.student_cnn,
                    "seed": config.seed,
                    "projector_type": projector_type,
                },
                config.output_dir / "last_checkpoint.pth",
            )

    if teacher_timing_events:
        torch.cuda.synchronize(device)
        teacher_forward_seconds = sum(
            start.elapsed_time(end) / 1000.0
            for start, end in teacher_timing_events
        )

    backbone_path = config.output_dir / "student_backbone.pth"
    torch.save(student.state_dict(), backbone_path)
    distillation_metadata = {
        "approach": "distillation",
        "projector_type": projector_type,
        "method": "cosine feature distillation",
        "teacher": "DINO ViT-S/16",
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_feature_dim": TEACHER_DIM,
        "teacher_input_size": TEACHER_SIZE,
        "teacher_normalization": "ImageNet mean/std",
        "augmentation": "VICRegTransform(input_size=64), first view shared by teacher/student",
        "student_cnn": config.student_cnn,
        "student_seed": config.seed,
        "student_parameter_count": parameter_count,
        "projector_parameter_count": sum(
            parameter.numel() for parameter in projector.parameters()
        ),
        "training_epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimizer_steps": config.epochs * len(train_loader),
        "teacher_forward_seconds": teacher_forward_seconds,
        "teacher_forward_device": str(device),
        "teacher_forward_gpu_hours": (
            teacher_forward_seconds / 3600.0 if device.type == "cuda" else None
        ),
        "student_backbone": str(backbone_path),
    }
    with (config.output_dir / "distillation.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(distillation_metadata, file, indent=2)
        file.write("\n")

    print(f"Saved student backbone to {backbone_path}")
    print(f"TensorBoard logs saved to {tensorboard_dir}")
    time_unit = "GPU-hours" if device.type == "cuda" else "CPU-hours"
    print(
        f"Teacher forward time: {teacher_forward_seconds:.1f}s "
        f"({teacher_forward_seconds / 3600.0:.4f} {time_unit})"
    )


def main():
    args = parse_args()
    minimum_seeds = 3 if args.student_cnn == "small" else 2
    if len(args.seeds) < minimum_seeds:
        raise ValueError(
            f"The {args.student_cnn} student requires at least {minimum_seeds} seeds."
        )
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain unique nonnegative integers.")

    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.teacher_checkpoint, device)
    size_name = "small_cnn" if args.student_cnn == "small" else "larger_cnn"
    run_root = args.output_dir
    if args.projector_type == "linear":
        run_root = run_root / "ablation" / "projector_linear"

    for seed in args.seeds:
        config = replace(
            base_config,
            experiment_name=f"distillation_{args.projector_type}_{size_name}",
            student_cnn=args.student_cnn,
            approach="distillation",
            seed=seed,
            output_dir=run_root / size_name / f"seed{seed}",
        )
        print(
            f"\nStarting distillation: projector={args.projector_type}, "
            f"student={args.student_cnn}, seed={seed}"
        )
        train_one(config, teacher, args.teacher_checkpoint, args.projector_type)


if __name__ == "__main__":
    main()
