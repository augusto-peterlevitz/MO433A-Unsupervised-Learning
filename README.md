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
data-loading workers. The saved `metadata.json` records the checkpoint's
resolved path, filename, SHA-256 digest, DINO source repository and paper,
pretraining dataset, feature pooling, and published compute references.

The official [DINO repository](https://github.com/facebookresearch/dino)
describes ViT-S/16 training on ImageNet and reports two compute references:
the 100-epoch run took 1.75 days on 8 GPUs (about 336 GPU-days), and the
300-epoch boosted run took 2.6 days on 16 GPUs (about 998 GPU-days). These
are recorded as published references, not as confirmed costs for the local
checkpoint: the checkpoint's exact training run has not been verified. In a
report, identify this uncertainty rather than attributing either reference
directly to the checkpoint. The DINO training data is ImageNet-1k used
self-supervised, without labels. The feature cache uses the model's CLS-token
output.

## Experiment Approach Names
Run and feature-cache metadata use `approach` identifiers instead of row letters:

| Project row | `approach` |
|---|---|
| A: random initialization | `random_baseline` |
| B: direct VICReg training | `vicreg` |
| C: DINO-to-student distillation | `distillation` |
| D: frozen teacher encoding | `teacher_encoding` |

The row letters are retained below as project-protocol terminology. Use the
approach identifier in configs, feature metadata, and CLI options that select
an experiment.

## Row B: VICReg Student Training
Run the direct-training baseline for one named variant of the project's
`StudentCNN` architecture. Specify all seeds for that variant with `--seeds`:

```bash
python3 -m scripts.train_student_cnn \
  --config configs/vicreg.json \
  --student-cnn small \
  --seeds 42 43 44 \
  --output-dir checkpoints/vicreg
```

This runs three seeds for the smaller variant. For the larger variant, run:

```bash
python3 -m scripts.train_student_cnn \
  --config configs/vicreg.json \
  --student-cnn large \
  --seeds 42 43 \
  --output-dir checkpoints/vicreg
```

Both use VICReg's two-view augmentation and loss, train on the unlabeled
training split, and use the `StudentCNN` backbone from `models/student_cnn.py`
with no classifier head. Hyperparameters such as epochs, batch size, learning
rate, and data paths come from the base config. Each run saves its config,
checkpoint, backbone weights, and TensorBoard logs under
`checkpoints/vicreg/{small_cnn,larger_cnn}/seed<seed>/`. Use
`--student-cnn small` and `--student-cnn large` in separate invocations. These
select the explicit `SmallStudent` and `LargerStudent` classes, respectively.
The project protocol requires at least three seeds for `small` and two for
`large`.

Extract features for all seeds in one command so the probe can report
seed-to-seed variation:

```bash
python3 -m probing.extract_features \
  --config configs/vicreg.json \
  --student-cnn small \
  --seeds 42 43 44
```

Use `--student-cnn large --seeds 42 43` for the larger variant. For a
non-default checkpoint location, pass `--checkpoint-root`; `--output-dir`
sets the common feature-cache root.

The standalone one-run trainer remains available for development smoke runs;
its architecture is selected by `student_cnn` in the config:

```bash
python3 -m training.train_vicreg --config configs/vicreg.json
```

## Row C: DINO-to-Student Distillation
Train the same student architectures from a frozen DINO ViT-S/16 teacher.
The config supplies the data paths and training budget; use the same settings
as row B for a budget-matched comparison:

```bash
python3 -m training.train_distillation \
  --config configs/vicreg.json \
  --student-cnn small \
  --seeds 42 43 44 \
  --teacher-checkpoint dino_deitsmall16_pretrain.pth \
  --output-dir checkpoints/distillation
```

For the larger student, use `--student-cnn large --seeds 42 43`. Each run
freezes the DINO teacher, gives the teacher and student the same VICReg
augmented view (student at 64px, teacher resized to 224px), and minimizes cosine
distance between the teacher features and an MLP projection of the student
features. Labels are ignored. It saves the student backbone, per-epoch loss,
configuration, teacher-forward timing, and TensorBoard event logs under
`checkpoints/distillation/{small_cnn,larger_cnn}/seed<seed>/`. Open the logs
for one run with:

```bash
tensorboard --logdir checkpoints/distillation/small_cnn/seed42/tensorboard
```

TensorBoard records batch and epoch distillation loss, learning rate, and epoch
duration. Each seed has its own timestamped run directory under `tensorboard/`.

Extract all seeds from the distilled row with:

```bash
python3 -m probing.extract_features \
  --config configs/vicreg.json \
  --approach distillation \
  --student-cnn small \
  --seeds 42 43 44
```

This reads the `distillation` checkpoints by default and stores features under
`datasets/features/distillation/small_cnn/seed<seed>/`. Use
`--student-cnn large --seeds 42 43` for the larger variant. The teacher is
loaded from the local checkpoint; DINO repository code must be available to
`torch.hub` on the first run.

## VICReg Feature Extraction
Extract deterministic backbone features, labels, and original dataset indices
for both the configured training split and the held-out test split:

```bash
python3 -m probing.extract_features
```

The default config is `configs/vicreg.json`. The script reads the dataset root,
training-subset pickle, model architecture, batch size, worker count, and checkpoint
location from that config. It uses `datasets/eurosat_test_subset.pkl` for the
test indices, loads `student_backbone.pth` from the configured output
directory, and writes the cache to
`datasets/features/vicreg/<small_cnn|larger_cnn>/seed<seed>`.

To override paths or extraction settings:

```bash
python3 -m probing.extract_features \
  --config checkpoints/vicreg/small_cnn/seed42/vicreg.json \
  --checkpoint-path checkpoints/vicreg/small_cnn/seed42/student_backbone.pth \
  --output-dir datasets/features/vicreg/small_cnn/seed42 \
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
  --student-cnn small \
  --seeds 42 43 44 \
  --batch-size 64 \
  --num-workers 4
```

This creates one deterministic feature cache per `(student variant, seed)` under
`datasets/features/random_baseline/{small_cnn,larger_cnn}/seed<seed>`. The two
architectures are selected with `--student-cnn small` and `--student-cnn large`.
The models are randomly
initialized, receive no pretraining, and are kept in `eval()` mode during
feature extraction. Keeping BatchNorm in `eval()` uses its default running
statistics and avoids making the extracted features depend on the extraction
batch composition. Run a separate command with `--student-cnn large --seeds 42 43` for the larger
variant. Preserve per-seed results and report individual runs for the two-seed
variant.

## SVM Probing and Hyperparameter Tuning
Tune the SVM parameters once on the direct-training (row B) VICReg features,
using only the full labeled training split. The grid search uses 5-fold
stratified CV and searches a small C/gamma grid for the RBF kernel:

```bash
python3 -m probing.svm_gridsearch \
  --features-dir datasets/features/vicreg/small_cnn/seed42 \
  --kernel rbf \
  --seed 42 \
  --output-dir runs/probes
```

The grid search uses one designated VICReg cache and saves the fixed parameters
under `runs/probes/svm_gridsearch/vicreg/<student-size>/` (including a seed
directory when the cache identifies one). `--seed` controls the cross-validation
split and `LinearSVC` randomness. Repeat with `--kernel linear` for the linear SVM.
Use the same parameter file for every seed and encoder; do not tune separately
for each seed.

```bash
python3 -m probing.svm_evaluate \
  --features-dir \
    datasets/features/random_baseline/small_cnn/seed42 \
    datasets/features/random_baseline/small_cnn/seed43 \
    datasets/features/random_baseline/small_cnn/seed44 \
  --params runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json \
  --output-dir runs/probes
```

`probing.svm_evaluate` supports both `linear` and `rbf` parameter files and
reports test accuracy at 1%, 10%, 50%, and 100% of the labeled training set,
for each draw in the training-subset pickle. It saves per-seed results and
summaries under `runs/probes/svm_evaluation/<approach>/<student-size>/<seed>/<kernel>/`
and pooled seed/draw results under
`runs/probes/svm_evaluation/<approach>/<student-size>/aggregate/<kernel>/`. The pooled
summary reports the mean and standard deviation across seeds and labeled
subset draws. The repeated identical 100% draws are counted once per seed in
the pooled summary. Parameter counts and cost metadata are read from
`metadata.json` when present. Do not tune on test data or separately for each
seed or encoder. Use the JSON file produced by `svm_gridsearch.py` for the
matching kernel. Run
`python3 -m probing.svm_evaluate --help` for the evaluator options.

## SVM Label-Efficiency Curve
After evaluating all desired seeds, create a log-scale curve from the nested
per-seed and aggregate summaries:

```bash
python3 -m probing.svm_evaluate \
  --features-dir \
    datasets/features/random_baseline/small_cnn/seed42 \
    datasets/features/random_baseline/small_cnn/seed43 \
    datasets/features/random_baseline/small_cnn/seed44 \
  --params runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json \
  --output-dir runs/probes
```

Use the RBF parameter JSON in the evaluation command instead for RBF results.
Plot the nested summaries with:

```bash
python3 -m probing.plot_label_efficiency \
  --probe linear \
  --input-dir runs/probes/svm_evaluation
```

The standalone `probing.plot_label_efficiency` script supports
`--probe linear`, `--probe rbf`, or `--probe knn`.

## KNN Label-Efficiency Curve
The KNN evaluator writes per-seed and combined summaries. Run the plot script
to generate a log-scale curve from those summaries:

```bash
python3 -m probing.knn_evaluate \
  --features-dir datasets/features/vicreg/small_cnn/seed42 \
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
  --features-dir datasets/features/vicreg/small_cnn/seed42 \
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
