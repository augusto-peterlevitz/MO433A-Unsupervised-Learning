import json

def load_encoder_cost(feature_dir):
    metadata = {}
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)

    num_parameters = metadata.get("num_parameters")
    latency_ms = metadata.get("latency_ms")
    device = metadata.get("device")
    batch_size = metadata.get("batch_size")

    return {
        "num_parameters": num_parameters,
        "inference_latency_ms": latency_ms,
        "device": device,
        "batch_size": batch_size,
    }