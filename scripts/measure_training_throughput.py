import argparse
import csv
import pickle
import time
from pathlib import Path

import torch
from lightly.loss import VICRegLoss
from torch.utils.data import DataLoader, Subset

from data.dataloaders import UnlabeledDataset
from data.eurosat import EuroSAT
from models.student_cnn import build_student
from models.vicreg import VICRegModel
from training.train_distillation import load_teacher
from training.train_vicreg import load_config
from training.transforms import vicreg_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure augmented-data-loader, VICReg training-step, and DINO "
            "forward throughput."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/vicreg.json"))
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("dino_deitsmall16_pretrain.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/probes/cost_analysis/throughput.csv"),
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    return parser.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_step(step, batch_size, device, warmup_steps, timed_steps):
    for _ in range(warmup_steps):
        step()
    sync(device)

    started = time.perf_counter()
    for _ in range(timed_steps):
        step()
    sync(device)
    elapsed_seconds = time.perf_counter() - started
    return elapsed_seconds, batch_size * timed_steps / elapsed_seconds


def benchmark_loader(config, device, warmup_steps, timed_steps):
    with config.subsets_path.open("rb") as file:
        train_indices = pickle.load(file)[(1.0, 0)]

    dataset = Subset(
        UnlabeledDataset(
            EuroSAT(root=config.dataset_root, transform=vicreg_transform)
        ),
        list(map(int, train_indices)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )
    iterator = iter(loader)
    for _ in range(warmup_steps):
        next(iterator)

    started = time.perf_counter()
    for _ in range(timed_steps):
        next(iterator)
    elapsed_seconds = time.perf_counter() - started
    return elapsed_seconds, config.batch_size * timed_steps / elapsed_seconds


def benchmark_vicreg_step(student_cnn, config, device, warmup_steps, timed_steps):
    use_amp = device.type == "cuda"
    model = VICRegModel(build_student(student_cnn)).to(device).train()
    criterion = VICRegLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    first_view = torch.randn(
        config.batch_size, 3, 64, 64, device=device
    )
    second_view = torch.randn_like(first_view)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            first_projection = model(first_view)
            second_projection = model(second_view)
        loss = criterion(first_projection.float(), second_projection.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    elapsed_seconds, images_per_second = benchmark_step(
        step,
        config.batch_size,
        device,
        warmup_steps,
        timed_steps,
    )
    return elapsed_seconds, images_per_second


def benchmark_teacher(checkpoint, config, device, warmup_steps, timed_steps):
    use_amp = device.type == "cuda"
    teacher = load_teacher(checkpoint, device)
    images = torch.randn(config.batch_size, 3, 224, 224, device=device)

    def step():
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            teacher(images)

    return benchmark_step(
        step,
        config.batch_size,
        device,
        warmup_steps,
        timed_steps,
    )


def main():
    args = parse_args()
    if args.warmup_steps < 0 or args.timed_steps < 1:
        raise ValueError(
            "--warmup-steps must be nonnegative and --timed-steps positive."
        )
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"DINO teacher checkpoint not found: {args.teacher_checkpoint}"
        )

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_model = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    rows = []

    def add_measurement(
        stage,
        student_cnn,
        input_size,
        elapsed,
        throughput,
        data_source,
    ):
        is_loader = stage == "augmented_dataloader"
        precision = (
            "float32 CPU transforms"
            if is_loader
            else "amp_float16" if device.type == "cuda" else "float32"
        )
        rows.append({
            "stage": stage,
            "student_cnn": student_cnn,
            "batch_size": config.batch_size,
            "input_size": input_size,
            "num_workers": config.num_workers if is_loader else "",
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "elapsed_seconds": elapsed,
            "images_per_second": throughput,
            "precision": precision,
            "data_source": data_source,
            "device": "cpu" if is_loader else str(device),
            "gpu_model": gpu_model,
        })

    loader_seconds, loader_throughput = benchmark_loader(
        config,
        device,
        args.warmup_steps,
        args.timed_steps,
    )
    add_measurement(
        "augmented_dataloader",
        "",
        64,
        loader_seconds,
        loader_throughput,
        "EuroSAT with VICReg augmentation",
    )

    for student_cnn in ("small", "large"):
        step_seconds, step_throughput = benchmark_vicreg_step(
            student_cnn,
            config,
            device,
            args.warmup_steps,
            args.timed_steps,
        )
        add_measurement(
            "vicreg_training_step",
            student_cnn,
            64,
            step_seconds,
            step_throughput,
            "synthetic GPU-resident tensors",
        )

    teacher_seconds, teacher_throughput = benchmark_teacher(
        args.teacher_checkpoint,
        config,
        device,
        args.warmup_steps,
        args.timed_steps,
    )
    add_measurement(
        "dino_teacher_forward",
        "",
        224,
        teacher_seconds,
        teacher_throughput,
        "synthetic 224x224 tensors",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Device: {device} ({gpu_model}), batch size: {config.batch_size}")
    for row in rows:
        print(
            f"{row['stage']} {row['student_cnn']}: "
            f"{row['images_per_second']:.2f} images/s"
        )
    print(f"Saved throughput measurements to {args.output}")


if __name__ == "__main__":
    main()
