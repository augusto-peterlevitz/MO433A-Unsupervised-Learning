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
  --features VICReg=datasets/features/vicreg_width_1 \
  --params runs/probes/svm_gridsearch_rbf.json \
  --output runs/probes/rbf_results.csv
```

`probing.svm_evaluate` supports both `linear` and `rbf` parameter files and
reports test accuracy for each fraction/draw in the training-subset pickle.
Use the JSON file produced by `svm_gridsearch.py` for the matching kernel.
Keep the parameter file selected on row B fixed for every encoder; do not tune
on test data or separately for each encoder. Run
`python3 -m probing.svm_evaluate evaluate --help` for the evaluator options.

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
