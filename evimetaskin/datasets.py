# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DIAGNOSIS_TO_INDEX: Dict[str, int] = {
    "actinic keratosis": 0,
    "basal cell carcinoma": 1,
    "dermatofibroma": 2,
    "melanoma": 3,
    "nevus": 4,
    "pigmented benign keratosis": 5,
    "squamous cell carcinoma": 6,
    "vascular lesion": 7,
}

INDEX_TO_DIAGNOSIS = {v: k for k, v in DIAGNOSIS_TO_INDEX.items()}

SEX_TO_INDEX = {"female": 0, "male": 1, "unknown": 2}
SITE_TO_INDEX = {
    "anterior torso": 0,
    "head/neck": 1,
    "lower extremity": 2,
    "palms/soles": 3,
    "posterior torso": 4,
    "upper extremity": 5,
    "unknown": 6,
}
METADATA_DIM = 1 + len(SEX_TO_INDEX) + len(SITE_TO_INDEX)


def build_clinical_prompt(row: pd.Series, task: str = "binary") -> str:
    age = row.get("age_approx")
    sex = row.get("sex")
    site = row.get("anatom_site_general")
    age_text = "unknown age" if pd.isna(age) or str(age).strip() == "" else f"{age} years old"
    sex_text = "unknown sex" if pd.isna(sex) or str(sex).strip() == "" else str(sex)
    site_text = "unknown anatomical site" if pd.isna(site) or str(site).strip() == "" else str(site)
    if task == "binary":
        question = "Determine whether the skin lesion is benign or malignant."
    else:
        question = "Classify the skin lesion into its fine-grained disease category."
    return (
        f"Dermoscopy image of a skin lesion from a {age_text} {sex_text} patient, "
        f"located at {site_text}. {question}"
    )


def build_structured_metadata(row: pd.Series) -> np.ndarray:
    age = row.get("age_approx")
    if pd.isna(age) or str(age).strip() == "":
        age_value = 0.0
    else:
        age_value = max(0.0, min(float(age), 100.0)) / 100.0

    sex = row.get("sex")
    sex_key = "unknown" if pd.isna(sex) or str(sex).strip() == "" else str(sex).strip().lower()
    sex_vec = np.zeros(len(SEX_TO_INDEX), dtype=np.float32)
    sex_vec[SEX_TO_INDEX.get(sex_key, SEX_TO_INDEX["unknown"])] = 1.0

    site = row.get("anatom_site_general")
    site_key = "unknown" if pd.isna(site) or str(site).strip() == "" else str(site).strip().lower()
    site_vec = np.zeros(len(SITE_TO_INDEX), dtype=np.float32)
    site_vec[SITE_TO_INDEX.get(site_key, SITE_TO_INDEX["unknown"])] = 1.0

    return np.concatenate([np.array([age_value], dtype=np.float32), sex_vec, site_vec]).astype(np.float32)


def default_image_transform(train: bool, image_size: int = 224):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ISIC2018Task3Dataset(Dataset):
    """Unified image + metadata dataset for ISIC2018 Task 3 classification.

    Returns fields used by both pure-vision baselines and later VLM adaptation:
    image, binary_label, multiclass_label, prompt, isic_id, diagnosis.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        task: str = "binary",
        transform=None,
        image_size: int = 224,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        if task not in {"binary", "multiclass"}:
            raise ValueError("task must be binary or multiclass")
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.image_dir = self.data_root / f"{split}-image" / "image"
        self.metadata_path = Path(metadata_path) if metadata_path else self.data_root / f"{split}-metadata.csv"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)
        required = {"isic_id", "target", "diagnosis"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

        self.df["image_path"] = self.df["isic_id"].map(lambda x: self.image_dir / f"{x}.jpg")
        self.df = self.df[self.df["image_path"].map(lambda p: p.exists())].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No matched images found for split={split} under {self.image_dir}")

        unknown = sorted(set(self.df["diagnosis"].dropna()) - set(DIAGNOSIS_TO_INDEX))
        if unknown:
            raise ValueError(f"Unknown diagnosis labels: {unknown}")

        self.transform = transform if transform is not None else default_image_transform(split == "train", image_size)

    def __len__(self):
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return 2 if self.task == "binary" else len(DIAGNOSIS_TO_INDEX)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        diagnosis = str(row["diagnosis"])
        return {
            "image": image,
            "binary_label": int(row["target"]),
            "multiclass_label": int(DIAGNOSIS_TO_INDEX[diagnosis]),
            "prompt": build_clinical_prompt(row, task=self.task),
            "metadata": build_structured_metadata(row),
            "isic_id": str(row["isic_id"]),
            "diagnosis": diagnosis,
        }
