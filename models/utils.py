import json

def load_encoder_cost(feature_dir):
    metadata = {}
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

    num_parameters = metadata.get("num_parameters")
    latency_ms = metadata.get(
        "inference_latency_ms",
        metadata.get("latency_ms"),
    )
    device = metadata.get("latency_device", metadata.get("device"))
    batch_size = metadata.get(
        "latency_batch_size",
        metadata.get("batch_size"),
    )

    return {
        "num_parameters": num_parameters,
        "flops_per_image": metadata.get("flops_per_image"),
        "gflops_per_image": metadata.get("gflops_per_image"),
        "inference_latency_ms": latency_ms,
        "inference_latency_std_ms": metadata.get("latency_std_ms"),
        "throughput_images_per_second": metadata.get(
            "throughput_images_per_second"
        ),
        "device": device,
        "gpu_model": metadata.get("gpu_model"),
        "batch_size": batch_size,
        "feature_batch_size": metadata.get("batch_size"),
        "latency_input_size": metadata.get("latency_input_size"),
        "latency_precision": metadata.get("latency_precision"),
    }