import time

import torch
from fvcore.nn import FlopCountAnalysis

from models.student_cnn import StudentCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, device, size=96, warmup=50, iterations=200):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(iterations):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    for name, width in [("small", 0.5), ("large", 1.0)]:
        model = StudentCNN(width_mult=width).to(device).eval()
        x = torch.randn(1, 3, 96, 96, device=device)

        flops = FlopCountAnalysis(model, x)
        total_flops = flops.total()

        latency = measure_latency(model, device)

        print(f"\n{name.upper()} STUDENT")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"FLOPs/image: {total_flops:,.0f}")
        print(f"GFLOPs/image: {total_flops / 1e9:.6f}")
        print(f"Latency (batch size 1): {latency:.3f} ms/image")
        print(f"Unsupported FLOP operators: {flops.unsupported_ops()}")


if __name__ == "__main__":
    main()