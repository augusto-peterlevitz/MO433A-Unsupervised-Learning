"""
Full VICReg evaluation for the MO433 project.

Runs:
  - k-NN (cosine) at 1%, 10%, 50%, 100%, for all three draws
  - Linear SVM at the same fractions/draws
  - RBF SVM at the same fractions/draws
  - SVM hyperparameter tuning once on the 100% VICReg training subset
  - k-means NMI/ARI on held-out test features
  - t-SNE visualization of held-out test features
  - parameter count and backbone inference latency

Run from the repository root:
python evaluation/evaluate_vicreg.py \
    --checkpoint checkpoints/vicreg_width_1.0/student_backbone.pth \
    --width-mult 1.0

The script expects:
  datasets/EuroSAT_RGB/
  datasets/eurosat_train_subsets.pkl
  datasets/eurosat_test_subset.pkl

Feature cache is rebuilt automatically if it is absent or incompatible.
"""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from data.eurosat import EuroSAT
from models.student_cnn import StudentCNN


FRACTIONS = (0.01, 0.10, 0.50, 1.00)
DRAWS = (0, 1, 2)
RBF_MAX_SAMPLES = 10_000
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Full VICReg representation evaluation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/vicreg_width_1.0/student_backbone.pth"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/EuroSAT_RGB"))
    parser.add_argument(
        "--train-subsets",
        type=Path,
        default=Path("datasets/eurosat_train_subsets.pkl"),
    )
    parser.add_argument(
        "--test-indices",
        type=Path,
        default=Path("datasets/eurosat_test_subset.pkl"),
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results/vicreg_width_1.0"),
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Ignore any existing feature cache and extract features again.",
    )
    return parser.parse_args()


def load_indices(train_subsets_path, test_indices_path):
    with open(train_subsets_path, "rb") as f:
        subsets = pickle.load(f)
    with open(test_indices_path, "rb") as f:
        test_indices = np.asarray(pickle.load(f), dtype=np.int64)

    # This is the canonical 100% training split used by the project.
    train_indices = np.asarray(subsets[(1.0, 0)], dtype=np.int64)

    # Check that all requested subset indices belong to the canonical train split.
    train_index_set = set(train_indices.tolist())
    for key, indices in subsets.items():
        missing = set(np.asarray(indices, dtype=np.int64).tolist()) - train_index_set
        if missing:
            raise ValueError(f"Subset {key} contains indices outside the 100% train split.")

    return subsets, train_indices, test_indices


def make_eval_transform():
    # Must match the deterministic evaluation preprocessing used for all encoders.
    return transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@torch.inference_mode()
def extract_features(backbone, loader, device):
    backbone.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device, non_blocking=True)
        features = backbone(images)
        all_features.append(features.float().cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def load_or_extract_features(
    args, backbone, dataset, train_indices, test_indices, device
):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "features_full.npz"

    if cache_path.exists() and not args.force_extract:
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {
                "train_features",
                "train_labels",
                "train_indices",
                "test_features",
                "test_labels",
                "test_indices",
            }
            if required.issubset(cache.files):
                cached_train_indices = cache["train_indices"].astype(np.int64)
                cached_test_indices = cache["test_indices"].astype(np.int64)
                if (
                    np.array_equal(cached_train_indices, train_indices)
                    and np.array_equal(cached_test_indices, test_indices)
                ):
                    print(f"Loading compatible feature cache: {cache_path}")
                    return (
                        cache["train_features"],
                        cache["train_labels"],
                        cache["test_features"],
                        cache["test_labels"],
                    )
        print("Existing feature cache is incompatible; extracting features again.")

    eval_dataset = EuroSAT(root=args.dataset_root, transform=make_eval_transform())
    train_loader = DataLoader(
        Subset(eval_dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print("Extracting training-split features...")
    train_features, train_labels = extract_features(backbone, train_loader, device)
    print("Extracting test-split features...")
    test_features, test_labels = extract_features(backbone, test_loader, device)

    np.savez_compressed(
        cache_path,
        train_features=train_features,
        train_labels=train_labels,
        train_indices=train_indices,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
    )
    print(f"Feature cache saved to: {cache_path}")
    return train_features, train_labels, test_features, test_labels


def make_linear_pipeline(c_value):
    return Pipeline(
        [
            ("l2_normalize", Normalizer(norm="l2")),
            ("standardize", StandardScaler()),
            (
                "svm",
                LinearSVC(C=c_value, random_state=SEED, max_iter=20_000),
            ),
        ]
    )


def make_rbf_pipeline(c_value, gamma_value):
    return Pipeline(
        [
            ("l2_normalize", Normalizer(norm="l2")),
            ("standardize", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma_value)),
        ]
    )


def fixed_stratified_cap(features, labels, max_samples=RBF_MAX_SAMPLES):
    """Return a deterministic, stratified sample if the current set is too large."""
    if len(labels) <= max_samples:
        return features, labels

    selected, _ = train_test_split(
        np.arange(len(labels)),
        train_size=max_samples,
        stratify=labels,
        random_state=SEED,
    )
    return features[selected], labels[selected]


def tune_svm_hyperparameters(train_features, train_labels):
    """
    Tune once on VICReg B at 100% labels.
    Scaling is inside each CV pipeline, so each fold fits its own scaler.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\nTuning Linear SVM C on the 100% training subset...")
    linear_search = GridSearchCV(
        estimator=make_linear_pipeline(1.0),
        param_grid={"svm__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    linear_search.fit(train_features, train_labels)
    best_linear_c = float(linear_search.best_params_["svm__C"])

    print("\nTuning RBF SVM C and gamma on a fixed stratified sample...")
    rbf_features, rbf_labels = fixed_stratified_cap(train_features, train_labels)
    # Three folds and a small grid keep the quadratic-cost RBF search practical.
    rbf_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rbf_search = GridSearchCV(
        estimator=make_rbf_pipeline(1.0, "scale"),
        param_grid={
            "svm__C": [0.1, 1.0, 10.0],
            "svm__gamma": ["scale", 0.01, 0.1],
        },
        scoring="accuracy",
        cv=rbf_cv,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    rbf_search.fit(rbf_features, rbf_labels)

    best = {
        "linear_svm_C": best_linear_c,
        "rbf_svm_C": float(rbf_search.best_params_["svm__C"]),
        "rbf_svm_gamma": rbf_search.best_params_["svm__gamma"],
        "linear_cv_best_accuracy": float(linear_search.best_score_),
        "rbf_cv_best_accuracy": float(rbf_search.best_score_),
        "linear_cv_folds": 5,
        "rbf_cv_folds": 3,
        "rbf_tuning_sample_count": int(len(rbf_labels)),
        "rbf_max_samples": RBF_MAX_SAMPLES,
        "tuning_encoder": "VICReg",
        "tuning_label_fraction": 1.0,
        "random_state": SEED,
    }
    return best


def measure_inference_latency(backbone, device, batch_size=128, warmup=10, repeats=50):
    backbone.eval()
    dummy = torch.randn(batch_size, 3, 64, 64, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = backbone(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            _ = backbone(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    batch_ms = (elapsed / repeats) * 1000.0
    return {
        "device": str(device),
        "batch_size": batch_size,
        "warmup_batches": warmup,
        "timed_batches": repeats,
        "milliseconds_per_batch": batch_ms,
        "milliseconds_per_image": batch_ms / batch_size,
    }


def evaluate_knn(x_train, y_train, x_test, y_test):
    min_per_class = int(np.unique(y_train, return_counts=True)[1].min())
    k = min(20, min_per_class)
    probe = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    probe.fit(x_train, y_train)
    predictions = probe.predict(x_test)
    return float(accuracy_score(y_test, predictions)), k


def evaluate_svm(x_train, y_train, x_test, y_test, svm_params, kind):
    if kind == "linear_svm":
        probe = make_linear_pipeline(svm_params["linear_svm_C"])
    elif kind == "rbf_svm":
        probe = make_rbf_pipeline(
            svm_params["rbf_svm_C"], svm_params["rbf_svm_gamma"]
        )
        x_train, y_train = fixed_stratified_cap(x_train, y_train)
    else:
        raise ValueError(f"Unknown SVM kind: {kind}")

    probe.fit(x_train, y_train)
    predictions = probe.predict(x_test)
    return float(accuracy_score(y_test, predictions)), len(y_train)


def evaluate_all_probes(
    subsets, train_indices, train_features, train_labels,
    test_features, test_labels, svm_params
):
    # Map original EuroSAT dataset index -> row in cached training feature matrix.
    row_for_original_index = {
        int(original_index): row
        for row, original_index in enumerate(train_indices)
    }

    results = []
    for fraction in FRACTIONS:
        for draw in DRAWS:
            key = (fraction, draw)
            if key not in subsets:
                raise KeyError(f"Missing labeled subset {key} in saved subset file.")

            original_subset = np.asarray(subsets[key], dtype=np.int64)
            rows = np.asarray(
                [row_for_original_index[int(i)] for i in original_subset],
                dtype=np.int64,
            )
            x_labeled = train_features[rows]
            y_labeled = train_labels[rows]

            print(
                f"\nEvaluating fraction={fraction:.0%}, draw={draw}, "
                f"labeled examples={len(y_labeled)}"
            )

            knn_accuracy, k = evaluate_knn(
                x_labeled, y_labeled, test_features, test_labels
            )
            results.append(
                {
                    "encoder": "VICReg",
                    "label_fraction": fraction,
                    "draw": draw,
                    "probe": "knn_cosine",
                    "accuracy": knn_accuracy,
                    "n_train_probe": len(y_labeled),
                    "knn_k": k,
                    "svm_C": "",
                    "svm_gamma": "",
                }
            )
            print(f"  k-NN (k={k}): {knn_accuracy:.4f}")

            linear_accuracy, linear_n = evaluate_svm(
                x_labeled, y_labeled, test_features, test_labels,
                svm_params, "linear_svm"
            )
            results.append(
                {
                    "encoder": "VICReg",
                    "label_fraction": fraction,
                    "draw": draw,
                    "probe": "linear_svm",
                    "accuracy": linear_accuracy,
                    "n_train_probe": linear_n,
                    "knn_k": "",
                    "svm_C": svm_params["linear_svm_C"],
                    "svm_gamma": "",
                }
            )
            print(f"  Linear SVM: {linear_accuracy:.4f}")

            rbf_accuracy, rbf_n = evaluate_svm(
                x_labeled, y_labeled, test_features, test_labels,
                svm_params, "rbf_svm"
            )
            results.append(
                {
                    "encoder": "VICReg",
                    "label_fraction": fraction,
                    "draw": draw,
                    "probe": "rbf_svm",
                    "accuracy": rbf_accuracy,
                    "n_train_probe": rbf_n,
                    "knn_k": "",
                    "svm_C": svm_params["rbf_svm_C"],
                    "svm_gamma": svm_params["rbf_svm_gamma"],
                }
            )
            print(f"  RBF SVM (n={rbf_n}): {rbf_accuracy:.4f}")

    return results


def evaluate_clustering(test_features, test_labels, n_classes):
    # K-means uses raw frozen test features; no labels enter the clustering fit.
    clusterer = KMeans(n_clusters=n_classes, random_state=SEED, n_init=10)
    cluster_ids = clusterer.fit_predict(test_features)
    return {
        "kmeans_nmi": float(normalized_mutual_info_score(test_labels, cluster_ids)),
        "kmeans_ari": float(adjusted_rand_score(test_labels, cluster_ids)),
        "n_clusters": int(n_classes),
    }


def save_tsne_plot(test_features, test_labels, class_names, output_path):
    print("\nComputing t-SNE projection of held-out test features...")
    # Fixed method and parameters must be reused for every encoder in the project.
    projection = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(test_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        c=test_labels,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="EuroSAT class", bbox_to_anchor=(1.02, 1))
    plt.title("VICReg test-set embeddings (t-SNE)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    subsets, train_indices, test_indices = load_indices(
        args.train_subsets, args.test_indices
    )

    dataset = EuroSAT(root=args.dataset_root, transform=None)

    backbone = StudentCNN(width_mult=args.width_mult).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()

    parameter_count = sum(parameter.numel() for parameter in backbone.parameters())
    print(f"Backbone parameters: {parameter_count:,}")
    #TODO: Check this
    latency = measure_inference_latency(backbone, device, batch_size=args.batch_size)
    print(
        "Inference latency: "
        f"{latency['milliseconds_per_batch']:.3f} ms/batch, "
        f"{latency['milliseconds_per_image']:.4f} ms/image"
    )

    train_features, train_labels, test_features, test_labels = load_or_extract_features(
        args, backbone, dataset, train_indices, test_indices, device
    )

    # Hyperparameters are tuned only once on VICReg's 100% training split.
    svm_params_path = args.output_dir / "svm_hyperparameters.json"
    if svm_params_path.exists():
        with open(svm_params_path, encoding="utf-8") as f:
            svm_params = json.load(f)
        print(f"Using previously selected SVM hyperparameters: {svm_params_path}")
    else:
        svm_params = tune_svm_hyperparameters(train_features, train_labels)
        with open(svm_params_path, "w", encoding="utf-8") as f:
            json.dump(svm_params, f, indent=2)
        print(f"SVM hyperparameters saved to: {svm_params_path}")

    results = evaluate_all_probes(
        subsets, train_indices, train_features, train_labels,
        test_features, test_labels, svm_params
    )

    results_csv = args.output_dir / "probe_results.csv"
    fieldnames = [
        "encoder", "label_fraction", "draw", "probe", "accuracy",
        "n_train_probe", "knn_k", "svm_C", "svm_gamma",
    ]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Aggregate mean and standard deviation across the three labeled-subset draws.
    summary = {}
    for fraction in FRACTIONS:
        summary[f"{fraction:.2f}"] = {}
        for probe_name in ("knn_cosine", "linear_svm", "rbf_svm"):
            values = [
                row["accuracy"] for row in results
                if row["label_fraction"] == fraction and row["probe"] == probe_name
            ]
            summary[f"{fraction:.2f}"][probe_name] = {
                "mean_accuracy": float(np.mean(values)),
                "std_accuracy": float(np.std(values, ddof=1)),
            }

    n_classes = len(dataset.classes)
    clustering = evaluate_clustering(test_features, test_labels, n_classes)
    normalized_test_features = test_features / np.clip(
        np.linalg.norm(test_features, axis=1, keepdims=True), a_min=1e-12, a_max=None
    )
    mean_embedding_std = float(np.std(normalized_test_features, axis=0).mean())
    tsne_path = args.output_dir / "tsne_vicreg.png"
    save_tsne_plot(test_features, test_labels, dataset.classes, tsne_path)

    metrics = {
        "encoder": "VICReg",
        "feature_dim": int(train_features.shape[1]),
        "parameter_count": int(parameter_count),
        "inference_latency": latency,
        "svm_hyperparameters": svm_params,
        "probe_summary_mean_std_over_draws": summary,
        "clustering": clustering,
        "mean_embedding_std_l2_normalized_test_features": mean_embedding_std,
        "evaluation_protocol": {
            "label_fractions": list(FRACTIONS),
            "draws": list(DRAWS),
            "knn_metric": "cosine",
            "knn_k_rule": "min(20, minimum class count in labeled subset)",
            "svm_preprocessing": "L2 Normalizer followed by StandardScaler fitted on training data",
            "rbf_max_training_examples": RBF_MAX_SAMPLES,
            "rbf_subsampling": "deterministic stratified sample, random_state=42",
            "test_split_used_for": [
                "final probe accuracy",
                "k-means NMI/ARI",
                "t-SNE visualization",
            ],
        },
    }
    metrics_path = args.output_dir / "full_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n========== FULL VICREG EVALUATION ==========")
    for fraction in FRACTIONS:
        print(f"\nLabel fraction: {fraction:.0%}")
        for probe_name, values in summary[f"{fraction:.2f}"].items():
            print(
                f"  {probe_name:12s}: "
                f"{values['mean_accuracy']:.4f} ± {values['std_accuracy']:.4f}"
            )
    print("\nClustering:")
    print(f"  k-means NMI: {clustering['kmeans_nmi']:.4f}")
    print(f"  k-means ARI: {clustering['kmeans_ari']:.4f}")
    print(f"  Mean embedding std: {mean_embedding_std:.6f}")
    print("\nSaved outputs:")
    print(f"  Probe results: {results_csv}")
    print(f"  Metrics:       {metrics_path}")
    print(f"  SVM params:    {svm_params_path}")
    print(f"  t-SNE plot:    {tsne_path}")
    print("============================================")


if __name__ == "__main__":
    main()
