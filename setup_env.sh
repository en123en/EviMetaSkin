#!/usr/bin/env bash
set -euo pipefail

python3.12 -m venv evivlm
source evivlm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python - <<'PY'
import torch
import torchvision
import evimetaskin

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("evimetaskin:", evimetaskin.__file__)
PY
