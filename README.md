# EviMetaSkin

Official implementation of **EviMetaSkin**, a lightweight metadata-guided evidential framework for malignant/non-malignant skin lesion classification and uncertainty-guided case referral.

The released code contains only the components used by the paper: the EfficientNet-based image encoder, structured clinical metadata encoder, metadata-guided gated fusion, evidential classification, baselines, ablations, external validation, uncertainty analysis, calibration, Grad-CAM++, bootstrap confidence intervals, and decision-curve analysis. Legacy segmentation/VLM modules and BioClinicalBERT are not required.

## Repository structure

```text
EviMetaSkin/
  evimetaskin/             # Dataset, model, metrics, and baseline implementations
  scripts/                 # Training, evaluation, analysis, and figure scripts
  configs/                 # Human-readable experiment configurations
  data/README.md            # Dataset layout and label requirements
  results/reported/         # Compact manuscript tables, metrics, and final figures
  requirements.txt
  pyproject.toml
  setup_env.sh
```

## Environment

Experiments were run on Ubuntu 22.04 with Python 3.12.2, PyTorch 2.11.0+cu128, TorchVision 0.26.0+cu128, CUDA 12.8, and an NVIDIA RTX PRO 6000 Blackwell GPU.

```bash
cd ~/project/EviMetaSkin
bash setup_env.sh
source evivlm/bin/activate
```

For a manual installation:

```bash
python3.12 -m venv evivlm
source evivlm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The CUDA version reported by `nvidia-smi` is the driver's maximum supported version; the PyTorch wheel in this repository uses CUDA 12.8.

## Data preparation

Raw datasets are not included. Follow [data/README.md](data/README.md), then verify the cleaned ISIC2018 split and generate its statistics:

```bash
python scripts/prepare_isic2018.py --data-root "data/ISIC2018 Task3"
python scripts/generate_dataset_stats.py --data-root "data/ISIC2018 Task3"
```

Prepare PAD-UFES-20 for external validation:

```bash
python scripts/prepare_pad_ufes20.py \
  --pad-root "data/zr7vgbcyr2-1" \
  --output-dir "data/external_validation" \
  --extract-pad \
  --skip-ph2
```

## Train EviMetaSkin

The paper model uses EfficientNet-B0, 11-dimensional structured metadata, gated fusion, a 0.95/0.05 CE-evidential objective, balanced sampling, and validation-only threshold selection.

```bash
python scripts/train.py \
  --data-root "data/ISIC2018 Task3" \
  --task binary \
  --visual-backbone efficientnet_b0 \
  --fusion-mode gated \
  --run-name effb0_gatedmeta_ev005_seed2026 \
  --epochs 30 \
  --batch-size 16 \
  --balanced-sampler \
  --ce-weight 0.95 \
  --evidential-weight 0.05 \
  --threshold-selection recall_constraint \
  --min-recall 0.85 \
  --seed 2026
```

Outputs are saved under `results/experiments/evimetaskin/binary/<run-name>/` and include the checkpoint, history, test metrics, predictions, and threshold-selection record.

## Baselines

Use the same data split and evaluation protocol for each baseline:

```bash
python scripts/train_baselines.py --data-root "data/ISIC2018 Task3" --task binary --model resnet50       --epochs 30 --batch-size 16 --balanced-sampler --threshold-selection recall_constraint --min-recall 0.85 --run-name recall085_sampler
python scripts/train_baselines.py --data-root "data/ISIC2018 Task3" --task binary --model efficientnet_b0 --epochs 30 --batch-size 16 --balanced-sampler --threshold-selection recall_constraint --min-recall 0.85 --run-name recall085_sampler
python scripts/train_baselines.py --data-root "data/ISIC2018 Task3" --task binary --model efficientnet_b3 --epochs 30 --batch-size 16 --balanced-sampler --threshold-selection recall_constraint --min-recall 0.85 --run-name recall085_sampler
python scripts/train_baselines.py --data-root "data/ISIC2018 Task3" --task binary --model vit_b_16        --epochs 30 --batch-size 16 --balanced-sampler --threshold-selection recall_constraint --min-recall 0.85 --run-name recall085_sampler
python scripts/summarize_baselines.py --task binary
```

## Ablation studies

The PowerShell entry point runs the fixed-seed CE/EDL and fusion ablations, including image-only, concatenation, and gated variants:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ablation.ps1 -Python python -Seed 2026
```

Equivalent settings are documented in `configs/ablation.yaml`.

## External validation

The following command directly applies the ISIC2018-trained checkpoint to PAD-UFES-20. No PAD-UFES-20 training or fine-tuning is performed. Use `--threshold 0.5` for the fixed external-deployment threshold reported in the cross-domain comparison.

```bash
python scripts/evaluate_external.py \
  --manifest "data/external_validation/pad_ufes20_external_manifest.csv" \
  --run-dir "results/experiments/evimetaskin/binary/effb0_gatedmeta_ev005_seed2026" \
  --threshold 0.5 \
  --output-dir "results/experiments/evimetaskin/binary/effb0_gatedmeta_ev005_seed2026/external_validation/pad_ufes20_thr050"
```

Baseline checkpoints can be evaluated with `scripts/evaluate_external_baselines.py` using the same manifest and fixed threshold.

## Reliability and clinical-value analyses

```bash
RUN="results/experiments/evimetaskin/binary/effb0_gatedmeta_ev005_seed2026"

python scripts/analyze_uncertainty.py --run-dir "$RUN" --coverages 1.0,0.9,0.8,0.7
python scripts/bootstrap_ci.py --predictions "$RUN/test_predictions.csv"
python scripts/calibration.py --predictions "$RUN/test_predictions.csv" --model-name EviMetaSkin --dataset-name ISIC2018
python scripts/gradcampp.py --run-dir "$RUN" --data-root "data/ISIC2018 Task3"
```

Decision-curve analysis accepts repeated `Name::predictions.csv` arguments:

```bash
python scripts/decision_curve.py \
  --model "ResNet50::results/experiments/baselines/binary/resnet50/recall085_sampler/test_predictions.csv" \
  --model "EfficientNet-B0::results/experiments/baselines/binary/efficientnet_b0/recall085_sampler/test_predictions.csv" \
  --model "EfficientNet-B3::results/experiments/baselines/binary/efficientnet_b3/recall085_sampler/test_predictions.csv" \
  --model "Ours::$RUN/test_predictions.csv" \
  --output-dir "results/analysis/dca/isic_test"
```

## Reported results

`results/reported/` preserves compact metrics, tables, and publication figures from the completed study. Full checkpoints and raw experiment directories are excluded from Git. Deep-learning reruns can vary across hardware and software; use the documented seed and environment for the closest reproduction and report repeated-run statistics when extending the study.

## License and citation

The source code is released under the MIT License. Dataset licenses are controlled by the original providers. Citation information will be added after publication.
