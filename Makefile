PYTHON := python3
CONFIG := configs/vicreg.json
OUTPUT_CHECKPOINTS := outputs/checkpoints
OUTPUT_FEATURES := outputs/features
OUTPUT_RUNS := outputs/runs/probes
SVM_CV_SPLITS ?= 5
FEATURE_BATCH_SIZE ?= 64
FEATURE_NUM_WORKERS ?= 4
EUROSAT_ROOT := datasets/EuroSAT_RGB
TRAIN_PKL := datasets/eurosat_train_subsets.pkl
TEST_PKL := datasets/eurosat_test_subset.pkl
DINO_CHECKPOINT := dino_deitsmall16_pretrain.pth
DINO_WEIGHTS_URL := https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
EUROSAT_URL := https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
SEEDS_SMALL := 42 43 44
SEEDS_LARGE := 42 43

.PHONY: help docker-build docker-run tensorboard download-assets download-dino download-eurosat generate-subsets teacher-features random-baseline \
        vicreg-small vicreg-large train-vicreg \
        distillation-small distillation-large train-distillation \
        distillation-small-linear distillation-large-linear distillation-ablation \
        baseline-features vicreg-features distillation-features distillation-linear-features \
        svm-gridsearch svm-gridsearch-rbf svm-gridsearch-linear \
        knn-probes svm-rbf-probes svm-linear-probes teacher-probes probes \
        plot-label-efficiency-svm-linear plot-label-efficiency-svm-rbf \
        plot-label-efficiency-knn plot-distillation-gap plots diagnostics run-experiments \
        run-experiments-smoke clean

help:
	@echo "Available Make targets:"
	@echo "  docker-build              Build the project Docker image"
	@echo "  docker-run                Start an interactive container with all GPUs enabled"
	@echo "  tensorboard               Launch TensorBoard for training logs"
	@echo "  download-dino             Download pretrained DINO ViT-S/16 weights"
	@echo "  download-eurosat          Download and extract the EuroSAT RGB dataset"
	@echo "  download-assets           Download DINO weights and EuroSAT dataset"
	@echo "  generate-subsets          Generate EuroSAT training/test subset indices"
	@echo "  random-baseline           Train/extract random-initialization baselines (row A)"
	@echo "  train-vicreg              Train VICReg small and large students (row B)"
	@echo "  train-distillation        Train MLP-distilled small and large students (row C)"
	@echo "  distillation-ablation     Train linear-projector distillation ablation"
	@echo "  teacher-features          Extract frozen DINO teacher features (row D)"
	@echo "  baseline-features         Extract random-baseline features"
	@echo "  vicreg-features           Extract VICReg features"
	@echo "  distillation-features     Extract MLP-distillation features"
	@echo "  distillation-linear-features  Extract linear-projector features"
	@echo "  svm-gridsearch            Tune linear and RBF SVM parameters once on VICReg"
	@echo "  knn-probes                Run KNN probes for every encoder"
	@echo "  svm-rbf-probes                Run RBF-SVM probes for every encoder"
	@echo "  svm-linear-probes         Run linear-SVM probes for every encoder"
	@echo "  teacher-probes            Run KNN and SVM probes on the frozen teacher"
	@echo "  probes                    Run KNN and SVM probes"
	@echo "  diagnostics               Run k-means and t-SNE diagnostics"
	@echo "  plot-distillation-gap     Plot paired C-versus-B gaps by size and projector"
	@echo "  plot-label-efficiency-*   Plot linear-SVM, RBF-SVM, or KNN curves"
	@echo "  plots                     Plot all label-efficiency curves"
	@echo "  run-experiments           Run the complete experiment pipeline (considering the subset generation is already done)"
	@echo "  run-experiments-smoke     Run the full pipeline with 1 epoch and smoke outputs"
	@echo "  clean                     Remove generated files under outputs/"

docker-build:
	docker build -t mo433-unsupervised .

docker-run:
	docker run --gpus all --rm -it \
	  -v "$$PWD:/workspace" \
	  -w /workspace \
	  mo433-unsupervised

tensorboard:
	tensorboard --logdir outputs/checkpoints

download-dino:
	@if [ -f $(DINO_CHECKPOINT) ]; then echo "DINO weights already exist: $(DINO_CHECKPOINT)"; else mkdir -p $(@D); curl -L --fail --retry 3 -o $(DINO_CHECKPOINT) $(DINO_WEIGHTS_URL); fi

download-eurosat:
	@mkdir -p datasets
	@if [ -d $(EUROSAT_ROOT) ] && [ -n "$$(find $(EUROSAT_ROOT) -mindepth 1 -type d -print -quit)" ]; then echo "EuroSAT dataset already exists: $(EUROSAT_ROOT)"; else curl -L --fail --retry 3 -o datasets/EuroSAT.zip $(EUROSAT_URL); unzip -q -o datasets/EuroSAT.zip -d datasets; if [ -d datasets/2750 ] && [ ! -e $(EUROSAT_ROOT) ]; then ln -s 2750 $(EUROSAT_ROOT); fi; fi


download-assets: download-dino download-eurosat

generate-subsets:
	$(PYTHON) -m data.generate_subsets \
	  --eurosat-root $(EUROSAT_ROOT) \
	  --train-pkl $(TRAIN_PKL) \
	  --test-fraction 0.20 \
	  --test-pkl $(TEST_PKL)

random-baseline:
	$(PYTHON) -m training.random_baseline \
	  --eurosat-root $(EUROSAT_ROOT) \
	  --train-pkl $(TRAIN_PKL) \
	  --test-pkl $(TEST_PKL) \
	  --student-cnn small \
	  --seeds $(SEEDS_SMALL) \
	  --batch-size 64 --num-workers 4
	$(PYTHON) -m training.random_baseline \
	  --eurosat-root $(EUROSAT_ROOT) \
	  --train-pkl $(TRAIN_PKL) \
	  --test-pkl $(TEST_PKL) \
	  --student-cnn large \
	  --seeds $(SEEDS_LARGE) \
	  --batch-size 64 --num-workers 4

vicreg-small:
	$(PYTHON) -m training.train_vicreg \
	  --config $(CONFIG) --student-cnn small --seeds $(SEEDS_SMALL) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/vicreg

vicreg-large:
	$(PYTHON) -m training.train_vicreg \
	  --config $(CONFIG) --student-cnn large --seeds $(SEEDS_LARGE) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/vicreg

train-vicreg: vicreg-small vicreg-large

distillation-small:
	$(PYTHON) -m training.train_distillation \
	  --config $(CONFIG) --student-cnn small --seeds $(SEEDS_SMALL) \
	  --projector-type mlp --teacher-checkpoint $(DINO_CHECKPOINT) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/distillation

distillation-large:
	$(PYTHON) -m training.train_distillation \
	  --config $(CONFIG) --student-cnn large --seeds $(SEEDS_LARGE) \
	  --projector-type mlp --teacher-checkpoint $(DINO_CHECKPOINT) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/distillation

train-distillation: distillation-small distillation-large

distillation-small-linear:
	$(PYTHON) -m training.train_distillation \
	  --config $(CONFIG) --student-cnn small --seeds $(SEEDS_SMALL) \
	  --projector-type linear --teacher-checkpoint $(DINO_CHECKPOINT) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/distillation

distillation-large-linear:
	$(PYTHON) -m training.train_distillation \
	  --config $(CONFIG) --student-cnn large --seeds $(SEEDS_LARGE) \
	  --projector-type linear --teacher-checkpoint $(DINO_CHECKPOINT) \
	  --output-dir $(OUTPUT_CHECKPOINTS)/distillation

distillation-ablation: distillation-small-linear distillation-large-linear

baseline-features:
	$(PYTHON) -m training.random_baseline \
	  --eurosat-root $(EUROSAT_ROOT) --train-pkl $(TRAIN_PKL) --test-pkl $(TEST_PKL) \
	  --student-cnn small --seeds $(SEEDS_SMALL) \
	  --batch-size $(FEATURE_BATCH_SIZE) --num-workers $(FEATURE_NUM_WORKERS) \
	  --output-dir $(OUTPUT_FEATURES)/random_baseline
	$(PYTHON) -m training.random_baseline \
	  --eurosat-root $(EUROSAT_ROOT) --train-pkl $(TRAIN_PKL) --test-pkl $(TEST_PKL) \
	  --student-cnn large --seeds $(SEEDS_LARGE) \
	  --batch-size $(FEATURE_BATCH_SIZE) --num-workers $(FEATURE_NUM_WORKERS) \
	  --output-dir $(OUTPUT_FEATURES)/random_baseline

vicreg-features:
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach vicreg --student-cnn small --seeds $(SEEDS_SMALL) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/vicreg --output-dir $(OUTPUT_FEATURES)/vicreg
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach vicreg --student-cnn large --seeds $(SEEDS_LARGE) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/vicreg --output-dir $(OUTPUT_FEATURES)/vicreg

distillation-features:
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach distillation --student-cnn small --seeds $(SEEDS_SMALL) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/distillation --output-dir $(OUTPUT_FEATURES)/distillation
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach distillation --student-cnn large --seeds $(SEEDS_LARGE) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/distillation --output-dir $(OUTPUT_FEATURES)/distillation

distillation-linear-features:
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach distillation --student-cnn small --seeds $(SEEDS_SMALL) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/distillation/ablation/projector_linear \
	  --output-dir $(OUTPUT_FEATURES)/distillation
	$(PYTHON) -m evaluation.extract_features \
	  --config $(CONFIG) --approach distillation --student-cnn large --seeds $(SEEDS_LARGE) \
	  --checkpoint-root $(OUTPUT_CHECKPOINTS)/distillation/ablation/projector_linear \
	  --output-dir $(OUTPUT_FEATURES)/distillation

teacher-features:
	$(PYTHON) -m evaluation.dino_extract_features \
	  --eurosat-root $(EUROSAT_ROOT) \
	  --checkpoint-path $(DINO_CHECKPOINT) \
	  --output-dir $(OUTPUT_FEATURES)/dino_vits16 \
	  --batch-size $(FEATURE_BATCH_SIZE) \
	  --num-workers $(FEATURE_NUM_WORKERS)

svm-gridsearch-rbf:
	$(PYTHON) -m evaluation.svm_gridsearch \
	  --features-dir $(OUTPUT_FEATURES)/vicreg/small_cnn/seed42 \
	  --kernel rbf --seed 42 --n-splits $(SVM_CV_SPLITS) --output-dir $(OUTPUT_RUNS)

svm-gridsearch-linear:
	$(PYTHON) -m evaluation.svm_gridsearch \
	  --features-dir $(OUTPUT_FEATURES)/vicreg/small_cnn/seed42 \
	  --kernel linear --seed 42 --n-splits $(SVM_CV_SPLITS) --output-dir $(OUTPUT_RUNS)

svm-gridsearch: svm-gridsearch-rbf svm-gridsearch-linear

knn-probes:
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn small --seeds $(SEEDS_SMALL) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn large --seeds $(SEEDS_LARGE) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn small --seeds $(SEEDS_SMALL) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn large --seeds $(SEEDS_LARGE) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type mlp --seeds $(SEEDS_SMALL) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type mlp --seeds $(SEEDS_LARGE) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type linear --seeds $(SEEDS_SMALL) --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type linear --seeds $(SEEDS_LARGE) --output-dir $(OUTPUT_RUNS)

svm-rbf-probes: svm-gridsearch-rbf
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn small --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn large --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn small --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn large --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type mlp --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type mlp --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type linear --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type linear --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)

teacher-probes: teacher-features svm-gridsearch-rbf svm-gridsearch-linear
	$(PYTHON) -m evaluation.knn_evaluate --feature-dir $(OUTPUT_FEATURES) --approach teacher_encoding --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach teacher_encoding --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_rbf.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach teacher_encoding --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)

svm-linear-probes: svm-gridsearch-linear
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn small --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach random_baseline --student-cnn large --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn small --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach vicreg --student-cnn large --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type mlp --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type mlp --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn small --projector-type linear --seeds $(SEEDS_SMALL) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)
	$(PYTHON) -m evaluation.svm_evaluate --feature-dir $(OUTPUT_FEATURES) --approach distillation --student-cnn large --projector-type linear --seeds $(SEEDS_LARGE) --params $(OUTPUT_RUNS)/svm_gridsearch/vicreg/small_cnn/seed42/svm_gridsearch_linear.json --output-dir $(OUTPUT_RUNS)

probes: knn-probes svm-rbf-probes svm-linear-probes

plot-label-efficiency-svm-linear:
	$(PYTHON) -m evaluation.plot_label_efficiency --probe linear --input-dir $(OUTPUT_RUNS)/svm_evaluation --output-dir $(OUTPUT_RUNS)/label_efficiency

plot-label-efficiency-svm-rbf:
	$(PYTHON) -m evaluation.plot_label_efficiency --probe rbf --input-dir $(OUTPUT_RUNS)/svm_evaluation --output-dir $(OUTPUT_RUNS)/label_efficiency

plot-label-efficiency-knn:
	$(PYTHON) -m evaluation.plot_label_efficiency --probe knn --input-dir $(OUTPUT_RUNS) --output-dir $(OUTPUT_RUNS)/label_efficiency

plot-distillation-gap:
	$(PYTHON) -m evaluation.plot_distillation_gap --probe linear --input-dir $(OUTPUT_RUNS) --output-dir $(OUTPUT_RUNS)/distillation_gap
	$(PYTHON) -m evaluation.plot_distillation_gap --probe rbf --input-dir $(OUTPUT_RUNS) --output-dir $(OUTPUT_RUNS)/distillation_gap
	$(PYTHON) -m evaluation.plot_distillation_gap --probe knn --input-dir $(OUTPUT_RUNS) --output-dir $(OUTPUT_RUNS)/distillation_gap

plots: plot-label-efficiency-svm-linear plot-label-efficiency-svm-rbf plot-label-efficiency-knn plot-distillation-gap

diagnostics:
	$(PYTHON) -m evaluation.representation_diagnostics \
	  --features-dir $(OUTPUT_FEATURES)/random_baseline/small_cnn/seed42 \
	  --features-dir $(OUTPUT_FEATURES)/vicreg/small_cnn/seed42 \
	  --features-dir $(OUTPUT_FEATURES)/distillation/projector_mlp/small_cnn/seed42 \
	  --features-dir $(OUTPUT_FEATURES)/distillation/projector_linear/small_cnn/seed42 \
	  --features-dir $(OUTPUT_FEATURES)/dino_vits16 \
	  --test-pkl $(TEST_PKL) \
	  --output-dir $(OUTPUT_RUNS)/representation_diagnostics

run-experiments: teacher-features train-vicreg train-distillation distillation-ablation baseline-features vicreg-features distillation-features distillation-linear-features svm-gridsearch probes teacher-probes diagnostics plots

run-experiments-smoke:
	$(MAKE) run-experiments \
	  CONFIG=configs/smoke.json \
	  OUTPUT_CHECKPOINTS=outputs/checkpoints/smoke \
	  OUTPUT_FEATURES=outputs/features/smoke \
	  OUTPUT_RUNS=outputs/runs/smoke \
	  SVM_CV_SPLITS=2 \
	  FEATURE_BATCH_SIZE=16 \
	  FEATURE_NUM_WORKERS=0

clean:
	rm -rf outputs
