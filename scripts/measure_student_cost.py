import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from models.student_cnn import build_student
from training.train_distillation import load_teacher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure parameters, FLOPs, and inference latency for encoders."
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("dino_deitsmall16_pretrain.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/probes/cost_analysis/encoder_costs.csv"),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    return parser.parse_args()


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def measure_latency(model, inputs, device, warmup, iterations):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            events = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(inputs)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize(device)
            timings = [start.elapsed_time(end) for start, end in events]
        else:
            timings = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(inputs)
                timings.append((time.perf_counter() - start) * 1000)
    return statistics.mean(timings), statistics.stdev(timings)


def measure_encoder(
    name,
    model,
    input_size,
    args,
    device,
    gpu_model,
    gpu_memory_mib,
):
    inputs = torch.randn(
        args.latency_batch_size,
        3,
        input_size,
        input_size,
        device=device,
    )
    model = model.to(device).eval()
    flop_analysis = FlopCountAnalysis(model, inputs)
    flops_per_image = flop_analysis.total() / args.latency_batch_size
    latency_ms, latency_std_ms = measure_latency(
        model,
        inputs,
        device,
        args.warmup,
        args.iterations,
    )
    unsupported_ops = flop_analysis.unsupported_ops()
    return {
        "encoder": name,
        "num_parameters": count_parameters(model),
        "flops_per_image": int(flops_per_image),
        "gflops_per_image": flops_per_image / 1e9,
        "inference_latency_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "throughput_images_per_second": (
            args.latency_batch_size * 1000 / latency_ms
        ),
        "latency_batch_size": args.latency_batch_size,
        "input_size": input_size,
        "precision": "float32",
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "device": str(device),
        "gpu_model": gpu_model,
        "gpu_memory_mib": gpu_memory_mib,
        "unsupported_flop_operators": str(unsupported_ops),
    }


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 2:
        raise ValueError("--warmup must be nonnegative and --iterations at least 2.")
    if args.latency_batch_size < 1:
        raise ValueError("--latency-batch-size must be positive.")
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(
            f"DINO teacher checkpoint not found: {args.teacher_checkpoint}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_model = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    gpu_memory_mib = (
        torch.cuda.get_device_properties(device).total_memory / (1024**2)
        if device.type == "cuda"
        else None
    )
    print(f"Device: {device} ({gpu_model})")

    encoders = [
        ("small_cnn", build_student("small"), 96),
        ("larger_cnn", build_student("large"), 96),
        (
            "dino_vits16",
            load_teacher(args.teacher_checkpoint, device),
            224,
        ),
    ]
    rows = [
        measure_encoder(
            name,
            model,
            input_size,
            args,
            device,
            gpu_model,
            gpu_memory_mib,
        )
        for name, model, input_size in encoders
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['encoder']}: {row['num_parameters']:,} parameters, "
            f"{row['gflops_per_image']:.4f} GFLOPs/image, "
            f"{row['inference_latency_ms']:.3f} +/- "
            f"{row['latency_std_ms']:.3f} ms/image "
            f"(batch={args.latency_batch_size}, input={row['input_size']})"
        )
        if row["unsupported_flop_operators"] != "Counter()":
            print(
                f"  FLOP counter did not support: "
                f"{row['unsupported_flop_operators']}"
            )
    print(f"Saved encoder measurements to {args.output}")


if __name__ == "__main__":
    main()
