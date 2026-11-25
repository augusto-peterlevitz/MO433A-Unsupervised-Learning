# MO433A-Unsupervised-Learning

## Environment setup (optional)

Conda is optional. To use it directly on your host, create and activate the
`mo433` environment, then install the project requirements with pip:

```bash
conda create -n mo433 python=3.8.10
conda activate mo433
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Alternatively, build the Docker image to use the project without creating a
host Conda environment:

```bash
docker build -t mo433-unsupervised .
```

Run an interactive container with the repository mounted so downloaded datasets
and generated outputs persist on the host:

```bash
docker run --rm -it \
  -v "$PWD:/workspace" \
  -w /workspace \
  mo433-unsupervised
```

Inside the container, download the dataset and DINO weights and run the
experiment Make targets as usual:

```bash
make download-assets
make generate-subsets
make run-experiments
```

To check that the full pipeline is wired up without using the standard output
directories, run:

```bash
make run-experiments-smoke
```

This uses `configs/smoke.json` (one training epoch, batch size 16, and no data
loader workers), retains the seed counts required by the training scripts, and
reduces SVM cross-validation to two folds. It still processes the complete
dataset and runs all pipeline stages, including teacher extraction and
diagnostics, so it is a wiring/integration check rather than a quick unit test.
Checkpoints, feature caches, and reports are written under
`outputs/{checkpoints,features,runs}/smoke/`. Downloaded assets and generated
subset files are still required.

For GPU training, pass `--gpus all` to `docker run` and ensure the host has
NVIDIA Container Toolkit installed. CPU execution remains available without it.

## Download required assets
Download both the EuroSAT RGB dataset and the pretrained DINO ViT-S/16 weights
with:

```bash
make download-assets
```

To download only one asset:

```bash
make download-eurosat
make download-dino
```

The EuroSAT archive is downloaded into `datasets/EuroSAT.zip` and extracted
under `datasets/`. The Make target exposes the class folders at
`datasets/EuroSAT_RGB`. The DINO checkpoint is saved as
`dino_deitsmall16_pretrain.pth` in the repository root.


## Subset Generation
Generate labeled EuroSAT subset indices and save them to a pickle file:

```bash
python3 -m data.generate_subsets \
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

## Row D: Frozen Teacher Encoding
Extract the EuroSAT representation from the frozen, pretrained DINO ViT-S/16
teacher for use as the row-D ceiling and cost reference:

```bash
python3 -m evaluation.dino_extract_features \
  --eurosat-root datasets/EuroSAT_RGB \
  --checkpoint-path dino_deitsmall16_pretrain.pth \
  --output-dir outputs/features/dino_vits16 \
  --batch-size 64 \
  --num-workers 4
```

This loads the official DINO `dino_vits16` architecture through `torch.hub`,
restores the local checkpoint with strict state loading, freezes all teacher
parameters, and performs deterministic teacher-side preprocessing with
`Resize(256)`, `CenterCrop(224)`, and ImageNet normalization. It saves the
384-dimensional CLS-token features, labels, and dataset indices in
`outputs/features/dino_vits16/{features.npy,labels.npy,indices.npy}`. The cache
also contains `metadata.json` with the checkpoint path and SHA-256 digest, DINO
source and paper, pretraining dataset, feature pooling, feature dimension, and
published compute references.

Row D does not have student seeds; the same frozen teacher extraction is used
for every probe. Use the cache with the SVM and KNN evaluators and include it in
the representation-diagnostics comparison. Before distillation or reporting
row-D accuracy, sanity-check the teacher with k-NN and k-means as described in
the project protocol. Near-chance k-NN accuracy, near-zero embedding variation,
or a k-means NMI near zero can indicate incorrect weights, preprocessing,
pooling, or collapse and should be diagnosed before further experiments.

The teacher is an upper-size-and-cost reference, not a trained student. In
reports, distinguish its published pretraining compute from the local
checkpoint's exact training history and record teacher forward GPU-hours
separately for row C.

## Cost and efficiency reporting
After the feature caches and probe results exist, measure encoder cost,
throughput, and end-to-end training efficiency with:

```bash
make encoder-costs
make cost-analysis
```

`encoder-costs` measures student encoders at 96x96 and DINO at 224x224 on a
fixed device. The inference benchmark uses batch size 1, 50 warm-up iterations,
and 200 timed iterations in float32; it records parameters, fvcore FLOPs per
image, mean and standard-deviation latency, throughput, GPU model, and any
unsupported FLOP operators. The throughput benchmark uses the selected
training config and reports augmented-loader images/s, VICReg training-step
images/s for both student sizes, and DINO forward images/s at the training
batch size with AMP when CUDA is available. The model-step and teacher-forward
benchmarks use synthetic device-resident inputs to isolate compute; the loader
benchmark includes EuroSAT decoding and VICReg augmentations on CPU. These
measurements are written to
`cost_analysis/{encoder_costs,throughput}.csv`.

Measured costs are attached to feature metadata and probe result CSVs.
`batch_size` in the probe CSVs is the latency-measurement batch size;
`feature_batch_size` preserves the feature-extraction batch size. The latency
columns therefore describe the fixed benchmark settings, not the probe's
classifier batch size.

`cost-analysis` reads VICReg epoch timings from TensorBoard (or the saved
`training_metadata.json` for new runs), distillation epoch timings and
teacher-forward GPU time from the run metadata, and the linear-SVM aggregate
accuracies. It writes per-seed local training cost to
`cost_analysis/training_gpu_hours.csv` and accuracy-per-GPU-hour scenarios to
`cost_analysis/accuracy_per_gpu_hour.csv`. The latter reports both published
DINO pretraining references. The checkpoint's exact pretraining run is
unverified; each C scenario conservatively charges the full reference cost,
while also listing local distillation GPU-hours and teacher-forward time
separately. Do not treat either published reference as the confirmed cost of
this checkpoint. For historical VICReg runs without device metadata, the
report marks the training device as inferred from the corresponding feature
cache; new VICReg runs save the device and GPU model explicitly.

## Row B: VICReg Student Training
Run the direct-training baseline for one named variant of the project's
`StudentCNN` architecture. Specify all seeds for that variant with `--seeds`:

```bash
python3 -m training.train_vicreg \
  --config configs/vicreg.json \
  --student-cnn small \
  --seeds 42 43 44 \
  --output-dir outputs/checkpoints/vicreg
```

This runs three seeds for the smaller variant. For the larger variant, run:

```bash
python3 -m training.train_vicreg \
  --config configs/vicreg.json \
  --student-cnn large \
  --seeds 42 43 \
  --output-dir outputs/checkpoints/vicreg
```

Both use VICReg's two-view augmentation and loss, train on the unlabeled
training split, and use the `StudentCNN` backbone from `models/student_cnn.py`
with no classifier head. Hyperparameters such as epochs, batch size, learning
rate, and data paths come from the base config. Each run saves its config,
checkpoint, backbone weights, and TensorBoard logs under
`outputs/checkpoints/vicreg/{small_cnn,larger_cnn}/seed<seed>/`. Use
`--student-cnn small` and `--student-cnn large` in separate invocations. These
select the explicit `SmallStudent` and `LargerStudent` classes, respectively.
The project protocol requires at least three seeds for `small` and two for
`large`.

Extract features for all seeds in one command so the probe can report
seed-to-seed variation:

```bash
python3 -m evaluation.extract_features \
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
  --output-dir outputs/checkpoints/distillation
```

For the larger student, use `--student-cnn large --seeds 42 43`. Each run
freezes the DINO teacher, gives the teacher and student the same VICReg
augmented view (student at 64px, teacher resized to 224px), and minimizes cosine
distance between the teacher features and an MLP projection of the student
features. Labels are ignored. It saves the student backbone, per-epoch loss,
configuration, teacher-forward timing, and TensorBoard event logs under
`outputs/checkpoints/distillation/{small_cnn,larger_cnn}/seed<seed>/`. Open the logs
for one run with:

```bash
tensorboard --logdir outputs/checkpoints/distillation/small_cnn/seed42/tensorboard
```

TensorBoard records batch and epoch distillation loss, learning rate, and epoch
duration. Each seed has its own timestamped run directory under `tensorboard/`.

### Required projector ablation
The required ablation compares the main MLP projector with a linear projector,
while keeping the cosine loss, data, training budget, seeds, and probes fixed.
The existing Row C MLP runs are the default setting; train the linear setting
for both student sizes with the same seed protocol:

```bash
python3 -m training.train_distillation \
  --config configs/vicreg.json \
  --student-cnn small \
  --seeds 42 43 44 \
  --projector-type linear \
  --teacher-checkpoint dino_deitsmall16_pretrain.pth \
  --output-dir outputs/checkpoints/distillation

python3 -m training.train_distillation \
  --config configs/vicreg.json \
  --student-cnn large \
  --seeds 42 43 \
  --projector-type linear \
  --teacher-checkpoint dino_deitsmall16_pretrain.pth \
  --output-dir outputs/checkpoints/distillation
```

The linear runs are saved separately under
`outputs/checkpoints/distillation/ablation/projector_linear/{small_cnn,larger_cnn}/seed<seed>/`;
the default MLP runs remain under
`outputs/checkpoints/distillation/{small_cnn,larger_cnn}/seed<seed>/`. Extract
distillation features for both settings; the extractor reads each run's
`distillation.json` and separates caches by projector type:

```bash
python3 -m evaluation.extract_features \
  --config configs/vicreg.json \
  --approach distillation \
  --student-cnn small \
  --seeds 42 43 44

python3 -m evaluation.extract_features \
  --config configs/vicreg.json \
  --approach distillation \
  --student-cnn small \
  --seeds 42 43 44 \
  --checkpoint-root outputs/checkpoints/distillation/ablation/projector_linear
```

The default feature-cache roots are
`outputs/features/distillation/projector_mlp/<student-size>/seed<seed>/` and
`outputs/features/distillation/projector_linear/<student-size>/seed<seed>/`.
Repeat extraction with `--student-cnn large --seeds 42 43` for both projector
settings. The evaluator outputs also keep projector types in separate
directories. Use the same VICReg-tuned SVM parameter files and KNN settings
for both variants and for the direct-VICReg baseline. This tests whether the
MLP's extra nonlinear capacity helps align student and teacher features;
the expectation is that linear may underperform, especially for the smaller
student, but the result is empirical. Compare each projector setting against
the matching VICReg student size, and report the C-versus-B gap by projector
setting, student size, label fraction, and seed/subset spread.

For example, evaluate the small-student VICReg baseline and both distillation
projectors together with KNN; the evaluator keeps their output folders and
seed aggregates separate:

```bash
python3 -m evaluation.knn_evaluate \
  --feature-dir outputs/features \
  --approach vicreg \
  --student-cnn small \
  --seeds 42 43 44 \
  --output-dir outputs/runs/probes

python3 -m evaluation.knn_evaluate \
  --feature-dir outputs/features \
  --approach distillation \
  --student-cnn small \
  --projector-type mlp \
  --seeds 42 43 44 \
  --output-dir outputs/runs/probes

python3 -m evaluation.knn_evaluate \
  --feature-dir outputs/features \
  --approach distillation \
  --student-cnn small \
  --projector-type linear \
  --seeds 42 43 44 \
  --output-dir outputs/runs/probes
```

For SVM, pass the same VICReg-tuned parameter file:

```bash
python3 -m evaluation.svm_evaluate \
  --feature-dir outputs/features \
  --approach vicreg \
  --student-cnn small \
  --seeds 42 43 44 \
  --params outputs/runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json \
  --output-dir outputs/runs/probes

python3 -m evaluation.svm_evaluate \
  --feature-dir outputs/features \
  --approach distillation \
  --student-cnn small \
  --projector-type mlp \
  --seeds 42 43 44 \
  --params outputs/runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json \
  --output-dir outputs/runs/probes

python3 -m evaluation.svm_evaluate \
  --feature-dir outputs/features \
  --approach distillation \
  --student-cnn small \
  --projector-type linear \
  --seeds 42 43 44 \
  --params outputs/runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json \
  --output-dir outputs/runs/probes
```

Repeat with large-student seeds `42 43` for all three encoders. This yields
the required two-projector-setting-by-two-student-size result grid; compare
each distillation result against VICReg of the same size. `make plots` creates
the paired C-versus-B gap figures for linear SVM, RBF SVM, and k-NN from those
per-seed results:

```bash
python3 -m evaluation.plot_distillation_gap \
  --probe linear \
  --input-dir outputs/runs/probes \
  --output-dir outputs/runs/probes/distillation_gap
```

The parameter-count figure plots the paired mean C-minus-B accuracy gap with
standard-deviation error bars against student-backbone parameter count at each
labeled-data fraction. The companion ablation figure shows the same gaps by
projector setting and student size. Pairing uses the same seed and subset draw
for B and C; duplicate full-data draws are counted once per seed. The output
directory also contains paired observations and a summary CSV with the gap,
spread, paired observation count, seed count, and mean B/C accuracies.
Parameter counts refer to the evaluated student backbone; the distillation
projector is discarded before feature extraction.

Extract all seeds from the distilled row with:

```bash
python3 -m evaluation.extract_features \
  --config configs/vicreg.json \
  --approach distillation \
  --student-cnn small \
  --seeds 42 43 44
```

This reads the MLP `distillation` checkpoints by default and stores features under
`outputs/features/distillation/projector_mlp/small_cnn/seed<seed>/`. Use
`--student-cnn large --seeds 42 43` for the larger variant. The teacher is
loaded from the local checkpoint; DINO repository code must be available to
`torch.hub` on the first run.

## VICReg Feature Extraction
Extract deterministic backbone features, labels, and original dataset indices
for both the configured training split and the held-out test split:

```bash
python3 -m evaluation.extract_features
```

The default config is `configs/vicreg.json`. The script reads the dataset root,
training-subset pickle, model architecture, batch size, worker count, and checkpoint
location from that config. It uses `datasets/eurosat_test_subset.pkl` for the
test indices, loads `student_backbone.pth` from the configured output
directory, and writes the cache to
`outputs/features/vicreg/<small_cnn|larger_cnn>/seed<seed>`.

To override paths or extraction settings:

```bash
python3 -m evaluation.extract_features \
  --config outputs/checkpoints/vicreg/small_cnn/seed42/vicreg.json \
  --checkpoint-path outputs/checkpoints/vicreg/small_cnn/seed42/student_backbone.pth \
  --output-dir outputs/features/vicreg/small_cnn/seed42 \
  --batch-size 64 \
  --num-workers 4
```

The output directory contains `features.npy`, `labels.npy`, and `indices.npy`;
rows in these files correspond to the saved original dataset indices. The
cache includes both training and test samples, so use only training indices
when tuning and keep test indices held out for final evaluation. Features are
saved raw; preprocessing is deferred to the SVM pipeline so the scaler is
fitted separately inside each cross-validation fold. Run
`python3 -m evaluation.extract_features --help` for all options.

## Row A Random-Baseline Feature Extraction
Extract row-A features for each random initialization and student size:

```bash
python3 -m training.random_baseline \
  --eurosat-root datasets/EuroSAT_RGB \
  --train-pkl datasets/eurosat_train_subsets.pkl \
  --test-pkl datasets/eurosat_test_subset.pkl \
  --student-cnn small \
  --seeds 42 43 44 \
  --batch-size 64 \
  --num-workers 4
```

This creates one deterministic feature cache per `(student variant, seed)` under
`outputs/features/random_baseline/{small_cnn,larger_cnn}/seed<seed>`. The two
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
python3 -m evaluation.svm_gridsearch \
  --features-dir outputs/features/vicreg/small_cnn/seed42 \
  --kernel rbf \
  --seed 42 \
  --output-dir outputs/runs/probes
```

The grid search uses one designated VICReg cache and saves the fixed parameters
under `outputs/runs/probes/svm_gridsearch/vicreg/<student-size>/` (including a seed
directory when the cache identifies one). `--seed` controls the cross-validation
split and `LinearSVC` randomness. Repeat with `--kernel linear` for the linear SVM.
Use the same parameter file for every seed and encoder; do not tune separately
for each seed.

```bash
python3 -m evaluation.svm_evaluate \
  --feature-dir outputs/features \
  --approach random_baseline \
  --student-cnn small \
  --seeds 42 43 44 \
  --params outputs/runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json \
  --output-dir outputs/runs/probes
```

`evaluation.svm_evaluate` supports both `linear` and `rbf` parameter files and
reports test accuracy at 1%, 10%, 50%, and 100% of the labeled training set,
for each draw in the training-subset pickle. It saves per-seed results and
summaries under `outputs/runs/probes/svm_evaluation/<approach>/<student-size>/<seed>/<kernel>/`
and pooled seed/draw results under
`outputs/runs/probes/svm_evaluation/<approach>/<student-size>/aggregate/<kernel>/`. The pooled
summary reports the mean and standard deviation across seeds and labeled
subset draws. The repeated identical 100% draws are counted once per seed in
the pooled summary. Parameter counts and cost metadata are read from
`metadata.json` when present. Do not tune on test data or separately for each
seed or encoder. Use the JSON file produced by `svm_gridsearch.py` for the
matching kernel. Run
`python3 -m evaluation.svm_evaluate --help` for the evaluator options.

## SVM Label-Efficiency Curve
After evaluating all desired seeds, create a log-scale curve from the nested
per-seed and aggregate summaries:

```bash
python3 -m evaluation.svm_evaluate \
  --feature-dir outputs/features \
  --approach random_baseline \
  --student-cnn small \
  --seeds 42 43 44 \
  --params outputs/runs/probes/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json \
  --output-dir outputs/runs/probes
```

Use the RBF parameter JSON in the evaluation command instead for RBF results.
Plot the nested summaries with:

```bash
python3 -m evaluation.plot_label_efficiency \
  --probe linear \
  --input-dir outputs/runs/probes/svm_evaluation
```

The standalone `evaluation.plot_label_efficiency` script supports
`--probe linear`, `--probe rbf`, or `--probe knn`.

## KNN Label-Efficiency Curve
The KNN evaluator writes per-seed and combined summaries. Run the plot script
to generate a log-scale curve from those summaries:

```bash
python3 -m evaluation.knn_evaluate \
  --feature-dir outputs/features \
  --approach vicreg \
  --student-cnn small \
  --seeds 42 \
  --output-dir outputs/runs/probes
```

For row A, pass every seed cache in the same command to aggregate both random
initializations and labeled-subset draws:

```bash
python3 -m evaluation.knn_evaluate \
  --feature-dir outputs/features \
  --approach random_baseline \
  --student-cnn small \
  --seeds 42 43 44 \
  --output-dir outputs/runs/probes
```

Results are saved beneath `outputs/runs/probes/knn_evaluation/`. Row-A per-seed
results and summaries go into directories such as
`random_baseline/small_cnn/seed42/`; the pooled seed/draw results are in the
sibling `random_baseline/small_cnn/aggregate/` directory. The KNN probe uses
cosine distance and, for every labeled subset and draw, chooses
`k = min(20, minimum labeled examples per class)`. Feature L2 normalization
and StandardScaler preprocessing are fitted consistently within each probe
training split. To create a label-efficiency plot from the nested KNN summaries,
run `python3 -m evaluation.plot_label_efficiency --probe knn --input-dir outputs/runs/probes`.

## Representation Diagnostics
Run the Section 3.7 diagnostics on the held-out test features for every
encoder. Repeat `--features-dir` once per cache:

```bash
python3 -m evaluation.representation_diagnostics \
  --features-dir outputs/features/vicreg/small_cnn/seed42 \
  --features-dir outputs/features/dino_vits16 \
  --output-dir outputs/runs/probes/representation_diagnostics
```

The script uses the same held-out test indices for all encoders, L2-normalizes
the cached features, and runs k-means with one cluster per test class. It saves
NMI and ARI in `representation_metrics.csv`, t-SNE coordinates in
`tsne_coordinates.csv`, and a comparable panel in `tsne_projections.png`.
All encoders use the same t-SNE perplexity and random seed; override them with
`--perplexity` and `--random-state` if needed. Run
`python3 -m evaluation.representation_diagnostics --help` for all options.


## Makefile experiment pipeline
The repository provides Make targets that run the project steps while preserving
the `outputs/checkpoints`, `outputs/features`, and `outputs/runs/probes` layout.

### Docker targets

For running all experiments at once, follow these steps:

1. Build and run the environment:
```bash
make docker-build
make docker-run
```


Alternativally you can use conda environment (if not created):
  ```bash
  conda create -n mo433 python=3.8.10
  conda activate mo433
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  ```

2. Download the EuroSAT RGB dataset (if now downloaded):

```bash
make download-eurosat
```
 
3. Download the pretrained DINO ViT-S/16 weights (if now downloaded):

```bash
make download-dino
````

4. Create the subsets (if not created):

```bash
make generate-subsets
```

5. Run all the project experiments at once:

```bash
make run-experiments
```

Targets are independent so they can be rerun without executing unrelated work.  Run `make help` to list them. All existing module commands of the sections above remain valid.