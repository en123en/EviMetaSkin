param(
    [string]$Python = "python",
    [int]$Epochs = 30,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$Seed = 2026,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"

$cudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
$device = "cuda"
if ($cudaAvailable.Trim() -ne "1") {
    if (-not $AllowCpu) {
        throw "CUDA is not available in the selected Python environment. Pass -AllowCpu only if the long CPU run is intentional."
    }
    $device = "cpu"
}

$base = @(
    "scripts/train.py",
    "--data-root", "data/ISIC2018 Task3",
    "--task", "binary",
    "--architecture", "structured_metadata",
    "--visual-backbone", "efficientnet_b0",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--lr", "1e-4",
    "--weight-decay", "1e-4",
    "--image-size", "224",
    "--balanced-sampler",
    "--threshold-selection", "recall_constraint",
    "--min-recall", "0.85",
    "--device", $device,
    "--seed", $Seed
)

& $Python @base --fusion-mode visual_only --ce-weight 1.0 --evidential-weight 0.0 --prediction-mode softmax --run-name effb0_imageonly_ce_softmax_seed2026
if ($LASTEXITCODE -ne 0) { throw "Image-only CE run failed with exit code $LASTEXITCODE" }

& $Python @base --fusion-mode gated --ce-weight 1.0 --evidential-weight 0.0 --prediction-mode softmax --run-name effb0_gatedmeta_ce_softmax_seed2026
if ($LASTEXITCODE -ne 0) { throw "Gated-metadata CE run failed with exit code $LASTEXITCODE" }

& $Python @base --fusion-mode visual_only --ce-weight 0.95 --evidential-weight 0.05 --prediction-mode evidential --run-name effb0_imageonly_ev005_seed2026
if ($LASTEXITCODE -ne 0) { throw "Image-only EDL run failed with exit code $LASTEXITCODE" }

& $Python @base --fusion-mode concat --ce-weight 0.95 --evidential-weight 0.05 --prediction-mode evidential --run-name effb0_concatmeta_ev005_seed2026
if ($LASTEXITCODE -ne 0) { throw "Concatenation EDL run failed with exit code $LASTEXITCODE" }

& $Python @base --fusion-mode gated --ce-weight 0.95 --evidential-weight 0.05 --prediction-mode evidential --run-name effb0_gatedmeta_ev005_seed2026
if ($LASTEXITCODE -ne 0) { throw "Gated-metadata EDL run failed with exit code $LASTEXITCODE" }

& $Python scripts/summarize_core_ablation.py
if ($LASTEXITCODE -ne 0) { throw "Table 2 summarization failed with exit code $LASTEXITCODE" }
