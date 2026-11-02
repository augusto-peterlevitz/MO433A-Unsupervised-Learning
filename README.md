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
python3 -m scripts.train_vicreg
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

## VICReg Evaluation
Evaluate a trained VICReg backbone on the held-out EuroSAT test set using
`configs/evaluate_vicreg.json`:

```bash
python3 -m evaluation.evaluate_vicreg
```

Pass `--config path/to/config.json` to use another JSON configuration. The
configuration contains the dataset, split, checkpoint, and output paths, model
width, batch size, worker count, and `seed`. The seed controls Python, NumPy,
PyTorch, data-loader worker initialization, and k-means initialization.
Evaluation metrics and extracted features are saved under the configured output
directory.

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
