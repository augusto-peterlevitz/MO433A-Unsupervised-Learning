import json

def load_encoder_cost(feature_dir):
    metadata = {}
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

    return {
        "num_parameters": metadata.get("num_parameters"),
        "flops_per_image": metadata.get("flops_per_image"),
        "gflops_per_image": metadata.get("gflops_per_image"),
        "inference_latency_ms": metadata.get(
            "inference_latency_ms", metadata.get("latency_ms")
        ),
        "inference_latency_std_ms": metadata.get(
            "inference_latency_std_ms", metadata.get("latency_std_ms")
        ),
        "throughput_images_per_second": metadata.get(
            "throughput_images_per_second"
        ),
        "device": metadata.get("device"),
        "gpu_model": metadata.get("gpu_model"),
        "batch_size": metadata.get("batch_size"),
        "feature_batch_size": metadata.get("feature_batch_size"),
        "latency_input_size": metadata.get(
            "latency_input_size", metadata.get("input_size")
        ),
        "latency_precision": metadata.get(
            "latency_precision", metadata.get("precision")
        ),
    }
