import os
import json
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG")


def _list_images(image_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(image_dir):
        if name.endswith(_IMAGE_EXTS):
            files.append(os.path.join(image_dir, name))
    return sorted(files)


def _load_label_mapping(label_json_path: str) -> Dict[str, int]:
    """Load malignancy labels from the shared JSON format.

    Expected JSON format is a list of dicts:
    [
        {"filename": "xxx.png", "malignancy": 0 or 1, "tirads": int},
        ...
    ]
    Only the `malignancy` field is used here.
    """
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("label_json must be a list of dicts with keys: filename, malignancy, tirads")

    mapping: Dict[str, int] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        if not filename:
            continue
        malignancy = int(item.get("malignancy", -1))
        if malignancy == -1:
            # skip unlabeled samples for pure classification
            continue
        mapping[os.path.basename(filename)] = malignancy
    return mapping


class ThyroidClsDataset(Dataset):
    """Thyroid ultrasound binary classification dataset (benign vs malignant).

    Images are read from `image_dir`, labels are loaded from a JSON file
    following the same format as used in the multitask pipeline.
    """

    def __init__(self, image_dir: str, label_json: str, img_size: int = 224, mode: str = "train") -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_map = _load_label_mapping(label_json)

        # only keep images that have a valid malignancy label
        all_images = _list_images(image_dir)
        self.images: List[str] = []
        self.labels: List[int] = []
        for path in all_images:
            fname = os.path.basename(path)
            if fname in self.label_map:
                self.images.append(path)
                self.labels.append(self.label_map[fname])

        if len(self.images) == 0:
            raise RuntimeError(
                f"No labeled images found in '{image_dir}'. "
                f"Make sure filenames in JSON match the image files."
            )

        if mode == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.images[idx]
        label = self.labels[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        img_t = self.transform(img)
        # BCEWithLogitsLoss expects float labels for binary classification
        label_t = torch.tensor(float(label), dtype=torch.float32)

        return {
            "image": img_t,
            "label": label_t,
            "filename": os.path.basename(path),
        }

