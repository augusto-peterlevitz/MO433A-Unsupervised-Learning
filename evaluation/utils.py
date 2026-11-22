import json
import re
from pathlib import Path


def load_feature_metadata(feature_dir):
    metadata_path = feature_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def get_encoder_name(feature_dir, metadata=None):
    metadata = metadata if metadata is not None else load_feature_metadata(feature_dir)
    if metadata.get("encoder"):
        return str(metadata["encoder"])
    if feature_dir.name.startswith("seed") and feature_dir.name[4:].isdigit():
        return feature_dir.parent.name
    return feature_dir.name


def get_aggregate_encoder_name(encoder_name):
    if isinstance(encoder_name, Path):
        encoder_name = get_encoder_name(encoder_name)
    elif not isinstance(encoder_name, str):
        encoder_name = str(encoder_name)
    return re.sub(r"_seed\d+$", "", encoder_name)


def get_cache_seed(feature_dir, metadata):
    if metadata.get("seed") is not None:
        return metadata["seed"]
    for part in reversed(feature_dir.parts):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    return None


def get_approach(metadata):
    approach = metadata.get("approach")
    if approach is None and metadata.get("row"):
        approach = {
            "A": "random_baseline",
            "B": "vicreg",
            "C": "distillation",
            "D": "teacher_encoding",
        }.get(str(metadata["row"]).upper())
    return approach


def get_student_size(metadata, feature_dir):
    student_cnn = metadata.get("student_cnn")
    if student_cnn in {"small", "large"}:
        return "small_cnn" if student_cnn == "small" else "larger_cnn"
    for size_name in ("small_cnn", "larger_cnn"):
        if size_name in feature_dir.parts:
            return size_name
    return None


def resolve_feature_dirs(feature_dir, approach, seeds, student_cnn, projector_type):
    if approach == "teacher_encoding":
        if seeds:
            raise ValueError("--seeds is not used with --approach teacher_encoding.")
        if student_cnn is not None:
            raise ValueError("--student-cnn is not used with --approach teacher_encoding.")
        if projector_type is not None:
            raise ValueError("--projector-type is not used with --approach teacher_encoding.")
        if (feature_dir / "features.npy").is_file():
            return [feature_dir]
        candidate = feature_dir / "dino_vits16"
        return [candidate if candidate.is_dir() else feature_dir]

    if not seeds:
        raise ValueError(f"--seeds is required with --approach {approach}.")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain unique nonnegative integers.")
    if student_cnn not in {"small", "large"}:
        raise ValueError(f"--student-cnn is required with --approach {approach}.")
    if approach == "distillation" and projector_type not in {"mlp", "linear"}:
        raise ValueError("--projector-type mlp or linear is required for distillation.")

    size_name = "small_cnn" if student_cnn == "small" else "larger_cnn"
    if (feature_dir / "features.npy").is_file():
        if len(seeds) != 1:
            raise ValueError(
                "A seed-specific --feature-dir can be used with exactly one --seeds value."
            )
        return [feature_dir]

    if any((feature_dir / f"seed{seed}").is_dir() for seed in seeds):
        base_dir = feature_dir
    else:
        base_dir = feature_dir / approach
        if approach == "distillation":
            base_dir /= f"projector_{projector_type}"
        base_dir /= size_name

    return [base_dir / f"seed{seed}" for seed in seeds]


def get_result_directory(output_dir, feature_dir, metadata, aggregate_encoder, probe_name):
    parts = feature_dir.parts
    if "random_baseline" in parts:
        baseline_index = parts.index("random_baseline")
        if baseline_index + 1 >= len(parts):
            raise ValueError(
                f"Expected a student-size directory after random_baseline in {feature_dir}."
            )
        return output_dir / probe_name / "random_baseline" / parts[baseline_index + 1]

    approach = metadata.get("approach")
    if approach is None and metadata.get("row"):
        approach = {
            "A": "random_baseline",
            "B": "vicreg",
            "C": "distillation",
            "D": "teacher_encoding",
        }.get(str(metadata["row"]).upper())
    if approach:
        student_size = get_student_size(metadata, feature_dir)
        encoder_directory = student_size or aggregate_encoder
        result_dir = output_dir / probe_name / str(approach) / encoder_directory
        if approach == "distillation" and metadata.get("projector_type"):
            result_dir /= f"projector_{metadata['projector_type']}"
        return result_dir
    return output_dir / probe_name / aggregate_encoder
