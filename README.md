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
  --output-pkl label_subsets.pkl
```

Both arguments are optional. By default, the script reads from
`datasets/EuroSAT_RGB` and writes `datasets/eurosat_subsets.pkl` in the current directory.

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