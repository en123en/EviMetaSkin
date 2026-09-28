# Data

Raw images are not redistributed. Download each dataset from its official source and place it under this directory.

## ISIC2018 Task 3

The training scripts expect the cleaned split used in the paper:

```text
data/ISIC2018 Task3/
  train-image/image/*.jpg
  train-metadata.csv
  validation-image/image/*.jpg
  validation-metadata.csv
  test-image/image/*.jpg
  test-metadata.csv
```

Each metadata CSV must contain `isic_id`, `diagnosis`, `target`, `age_approx`, `sex`, and `anatom_site_general`. The binary target is `0` for non-malignant and `1` for malignant. The paper's cleaned labels split the original AKIEC-related records into actinic keratosis and squamous cell carcinoma without introducing additional images; consult the manuscript for provenance and mapping details.

## PAD-UFES-20

Place the official `metadata.csv` and image archives/directories under `data/zr7vgbcyr2-1/`, then run:

```bash
python scripts/prepare_pad_ufes20.py --skip-ph2 --extract-pad
```

This creates `data/external_validation/pad_ufes20_external_manifest.csv`. PAD-UFES-20 is used only for external evaluation; no fine-tuning or threshold optimization is performed on it.

All dataset licenses and terms remain with their original providers.
