# MO433A-Unsupervised-Learning

## Environment setup

Create and activate the `mo433` Conda environment, then install the project
requirements with pip:

```bash
conda create -n mo433 python=3.8.10
conda activate mo433
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```
## Subset Generation
Generate labeled EuroSAT subset indices and save them to a pickle file:

```bash
python3 -m scripts.generate_subsets \
  --eurosat-root datasets/EuroSAT_RGB \
  --train-pkl datasets/eurosat_train_subsets.pkl \
  --test-fraction 0.20 \
  --test-pkl datasets/eurosat_test_subset.pkl
```

All options are optional. By default, the script reads from
`datasets/EuroSAT_RGB`, holds out 20% of the samples for final testing, and
writes training subset indices to `datasets/eurosat_train_subsets.pkl` and held-out
test indices to `datasets/eurosat_test_subset.pkl`. The training subsets are
generated only from the remaining samples, so they do not overlap the test set.
Use `--test-fraction` to set a different test proportion; it must be between 0
and 1 (for example, `0.20` for 20%). Both pickle files contain dataset indices.

## DINO Feature Extraction
Extract DINO ViT-S/16 features and labels for EuroSAT:

```bash
python3 -m scripts.dino_extract_features \
  --eurosat-root datasets/EuroSAT_RGB \
  --checkpoint-path dino_deitsmall16_pretrain.pth \
  --output-dir datasets/features/dino_vits16
```

All arguments are optional. Defaults are `datasets/EuroSAT_RGB` for the dataset,
`dino_deitsmall16_pretrain.pth` for the checkpoint,
`datasets/features/dino_vits16` for output, a batch size of `64`, and `4`
data-loading workers.

## Supervised Student CNN Training
Train the student CNN with a linear classification head and a stratified
train/validation split:

```bash
python3 -m scripts.train_student_cnn \
  --eurosat-root datasets/EuroSAT_RGB \
  --epochs 30 \
  --batch-size 64 \
  --output-dir runs/student_cnn
```

The script logs train/validation loss and accuracy for each epoch to
`runs/student_cnn/metrics.csv` and saves the checkpoint with the best
validation accuracy to `runs/student_cnn/best_model.pt`. Use
`python3 -m scripts.train_student_cnn --help` to see all options.

## VICReg Training
Train the student backbone with VICReg using the settings in
`configs/vicreg.json`:

```bash
python3 -m training.train_vicreg --config configs/vicreg.json
```

Pass `--config path/to/config.json` to load a different JSON configuration.
The configuration contains an `experiment_name` (default `vicreg_training`),
dataset and subset paths, model width, training hyperparameters, random seed,
worker count, and output directory. The resolved configuration is saved as
`vicreg.json` in that output directory along with `last_checkpoint.pth` and
`student_backbone.pth`. Batch and epoch metrics are logged to a uniquely named
TensorBoard run directory under `tensorboard/`, using the experiment name and
start date/time. The name and timestamp are also recorded in TensorBoard. View
all runs with:

```bash
tensorboard --logdir checkpoints/vicreg_width_1.0/tensorboard
```

## VICReg Feature Extraction
Extract deterministic backbone features, labels, and original dataset indices
for both the configured training split and the held-out test split:

```bash
python3 -m probing.extract_features
```

The default config is `configs/vicreg.json`. The script reads the dataset root,
training-subset pickle, model width, batch size, worker count, and checkpoint
location from that config. It uses `datasets/eurosat_test_subset.pkl` for the
test indices, loads `student_backbone.pth` from the configured output
directory, and writes the cache to
`datasets/features/vicreg_width_<width_mult>`.

To override paths or extraction settings:

```bash
python3 -m probing.extract_features \
  --checkpoint-path checkpoints/vicreg_width_1.0/student_backbone.pth \
  --output-dir datasets/features/vicreg_width_1 \
  --batch-size 64 \
  --num-workers 4
```

The output directory contains `features.npy`, `labels.npy`, and `indices.npy`;
rows in these files correspond to the saved original dataset indices. The
cache includes both training and test samples, so use only training indices
when tuning and keep test indices held out for final evaluation. Features are
saved raw; preprocessing is deferred to the SVM pipeline so the scaler is
fitted separately inside each cross-validation fold. Run
`python3 -m probing.extract_features --help` for all options.

## Row A Random-Baseline Feature Extraction
Extract row-A features for each random initialization and student size:

```bash
python3 -m scripts.random_baseline \
  --eurosat-root datasets/EuroSAT_RGB \
  --train-pkl datasets/eurosat_train_subsets.pkl \
  --test-pkl datasets/eurosat_test_subset.pkl \
  --width-mults 0.5 1.0 \
  --seeds 42 43 44 \
  --batch-size 64 \
  --num-workers 4
```

This creates one deterministic feature cache per `(width, seed)` under
`datasets/features/random_baseline/{small_cnn,larger_cnn}/seed<seed>`. The two
widths are assigned to `small_cnn` and `larger_cnn` by ascending width. The
models are randomly
initialized, receive no pretraining, and are kept in `eval()` mode during
feature extraction. Keeping BatchNorm in `eval()` uses its default running
statistics and avoids making the extracted features depend on the extraction
batch composition. The default command covers two student widths and three
seeds; for the large-size protocol, run the larger width with at least two
seeds and report individual runs as well as mean and standard deviation.

## SVM Probing and Hyperparameter Tuning
Tune the SVM parameters once on the direct-training (row B) VICReg features,
using only the full labeled training split. The grid search uses 5-fold
stratified CV and searches a small C/gamma grid for the RBF kernel:

```bash
python3 -m probing.svm_gridsearch \
  --features-dir datasets/features/vicreg_width_1 \
  --kernel rbf \
  --output-dir runs/probes
```

This saves `svm_gridsearch_rbf.json` in `runs/probes`. Repeat with
`--kernel linear` for the linear SVM. Then evaluate the fixed
parameters on the same label subsets for each encoder; repeat `--features`
for every encoder cache:

```bash
python3 -m probing.svm_evaluate \
  --features-dir datasets/features/vicreg_width_1 \
  --params runs/probes/svm_gridsearch_rbf.json \
  --output-dir runs/probes
```

`probing.svm_evaluate` supports both `linear` and `rbf` parameter files and
reports test accuracy at 1%, 10%, 50%, and 100% of the labeled training set,
for each draw in the training-subset pickle. It writes per-draw results and a
mean/std summary by `(encoder, probe, fraction)`. The summary includes encoder
parameter count and inference-cost fields. Parameter counts and cost metadata
are read from `metadata.json` in the feature cache when present; they can also
be supplied with `--num-parameters`, `--latency-ms`, `--device`, and
`--batch-size`. Run it once per encoder with the same parameter file selected on
row B; do not tune on test data or separately for each encoder. Use the JSON
file produced by `svm_gridsearch.py` for the matching kernel. Run
`python3 -m probing.svm_evaluate --help` for the evaluator options.

## SVM Label-Efficiency Curve
The SVM evaluator automatically combines summary CSVs for the selected kernel
and generates a log-scale curve after each run:

```bash
python3 -m probing.svm_evaluate \
  --features-dir datasets/features/vicreg_width_1 \
  --params runs/probes/svm_gridsearch_linear.json \
  --output-dir runs/probes
```

Use the RBF parameter JSON instead to generate the RBF-SVM curve. Each output
directory gets kernel-specific files such as
`label_efficiency/label_efficiency_linear_curves.png` or
`label_efficiency/label_efficiency_rbf_curves.png`. Override the plot location
with `--plot-dir`, or disable plotting with `--no-plot`.

The standalone `probing.plot_label_efficiency` script supports
`--probe linear`, `--probe rbf`, or `--probe knn`.

## KNN Label-Efficiency Curve
The KNN evaluator writes per-seed and combined summaries. Run the plot script
to generate a log-scale curve from those summaries:

```bash
python3 -m probing.knn_evaluate \
  --features-dir datasets/features/vicreg_width_1 \
  --output-dir runs/probes
```

For row A, pass every seed cache in the same command to aggregate both random
initializations and labeled-subset draws:

```bash
python3 -m probing.knn_evaluate \
  --features-dir \
    datasets/features/random_baseline/small_cnn/seed42 \
    datasets/features/random_baseline/small_cnn/seed43 \
    datasets/features/random_baseline/small_cnn/seed44 \
  --output-dir runs/probes
```

Results are saved beneath `runs/probes/knn_evaluation/`. Row-A per-seed
results and summaries go into directories such as
`random_baseline/small_cnn/seed42/`; the pooled seed/draw results are in the
sibling `random_baseline/small_cnn/aggregate/` directory. The KNN probe uses
cosine distance and, for every labeled subset and draw, chooses
`k = min(20, minimum labeled examples per class)`. Feature L2 normalization
and StandardScaler preprocessing are fitted consistently within each probe
training split. To create a label-efficiency plot from the nested KNN summaries,
run `python3 -m probing.plot_label_efficiency --probe knn --input-dir runs/probes`.

## Representation Diagnostics
Run the Section 3.7 diagnostics on the held-out test features for every
encoder. Repeat `--features-dir` once per cache:

```bash
python3 -m probing.representation_diagnostics \
  --features-dir datasets/features/vicreg_width_1 \
  --features-dir datasets/features/dino_vits16 \
  --output-dir runs/probes/representation_diagnostics
```

The script uses the same held-out test indices for all encoders, L2-normalizes
the cached features, and runs k-means with one cluster per test class. It saves
NMI and ARI in `representation_metrics.csv`, t-SNE coordinates in
`tsne_coordinates.csv`, and a comparable panel in `tsne_projections.png`.
All encoders use the same t-SNE perplexity and random seed; override them with
`--perplexity` and `--random-state` if needed. Run
`python3 -m probing.representation_diagnostics --help` for all options.

## DataLoader Smoke Test
Build the VICReg, distillation, labeled-training, and final-test data loaders
and print their dataset sizes:

```bash
python3 -m scripts.test_dataloaders \
  --eurosat-root datasets/EuroSAT_RGB \
  --train-pkl datasets/eurosat_train_subsets.pkl \
  --test-pkl datasets/eurosat_test_subset.pkl
```

All options are optional. The defaults use the dataset and pickle paths shown
above. Use `python3 -m scripts.test_dataloaders --help` to see all options.
