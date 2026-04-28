## ResNet50 Thyroid Ultrasound Binary Classification

This sub-project provides a simple ResNet50-based binary classifier
for thyroid ultrasound images (benign vs malignant) on top of the
existing multitask project.

### Data format

- **Images**: stored in a directory, e.g. `path/to/train_images/`.
- **Labels JSON**: same format as used by the multitask code and
  pyradiomics scripts:

```json
[
  {
    "filename": "image_0001.png",
    "malignancy": 0,
    "tirads": 3
  },
  {
    "filename": "image_0002.png",
    "malignancy": 1,
    "tirads": 5
  }
]
```

Only the `malignancy` field is used for this binary classification
task (0=benign, 1=malignant). Samples with `malignancy == -1` are
ignored.

### Training

From the project root (`dino_unet_multitask`), run:

```bash
python train_resnet50_cls.py \
  --train_image_dir /path/to/train_images \
  --train_label_json /path/to/train_labels.json \
  --val_image_dir /path/to/val_images \
  --val_label_json /path/to/val_labels.json \
  --epoch 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --dataset_name thyroid_cls
```

Key arguments:

- `--train_image_dir`: directory of training images.
- `--train_label_json`: JSON with training labels (filename + malignancy).
- `--val_image_dir` / `--val_label_json`: optional validation set.
- `--epoch`, `--batch_size`, `--lr`: standard training hyperparameters.
- `--img_size`: input size (default 224).
- `--cuda_device`: GPU index (default 0).

Checkpoints are saved under:

- `checkpoints_resnet50/<dataset_name>/<timestamp>/`

Logs (including training / validation losses and metrics) are stored in:

- `logs_resnet50/<dataset_name>/<timestamp>/resnet50_<dataset_name>_log.log`

