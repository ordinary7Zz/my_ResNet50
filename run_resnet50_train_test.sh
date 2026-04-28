#!/usr/bin/env bash

set -e

# ===================== User configuration =====================

# Root directory for your thyroid ultrasound dataset on the Linux server
DATA_ROOT="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test"

# Training set
TRAIN_IMAGE_DIR="${DATA_ROOT}/Superimposed_multitask/dataset_3/train/images"
TRAIN_LABEL_JSON="${DATA_ROOT}/Superimposed_multitask/dataset_3/train/dataset_3_train_label.json"

# Test sets (one or more; metrics will be reported for each)
# Example with a single test set:
#   TEST_IMAGE_DIRS=( "${DATA_ROOT}/test_images" )
#   TEST_LABEL_JSONS=( "${DATA_ROOT}/test_labels.json" )
# Example with multiple test sets:
#   TEST_IMAGE_DIRS=( "${DATA_ROOT}/testA_images" "${DATA_ROOT}/testB_images" )
#   TEST_LABEL_JSONS=( "${DATA_ROOT}/testA_labels.json" "${DATA_ROOT}/testB_labels.json" )
#   TEST_DATASET_NAMES=( "testA" "testB" )

TEST_IMAGE_DIRS=( "${DATA_ROOT}/TN3K/test/images" "${DATA_ROOT}/TN5K/test/images" "${DATA_ROOT}/ThyroidXL/test/images" "${DATA_ROOT}/DDTI_Classification/all/images")
TEST_LABEL_JSONS=( "${DATA_ROOT}/TN3K/test/TN3K_test_label.json" "${DATA_ROOT}/TN5K/test/TN5K_test_label.json" "${DATA_ROOT}/ThyroidXL/test/ThyroidXL_test_label.json" "${DATA_ROOT}/DDTI_Classification/all/DDTI_Classification_test_label.json")
# Optional custom names for each test dataset (same length as TEST_IMAGE_DIRS)
TEST_DATASET_NAMES=( "TN3K" "TN5K" "ThyroidXL" "DDTI_Classification")

# Training hyper-parameters
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
IMG_SIZE=224
CUDA_DEVICE=0
DATASET_NAME="thyroid_cls_resnet50"

# ===================== Run training + test =====================

# NOTE: run this script from the project root directory:
#   cd /path/to/dino_unet_multitask
#   bash resnet50/run_resnet50_train_test.sh

python -u train_resnet50_cls.py \
  --train_image_dir "${TRAIN_IMAGE_DIR}" \
  --train_label_json "${TRAIN_LABEL_JSON}" \
  --test_image_dirs "${TEST_IMAGE_DIRS[@]}" \
  --test_label_jsons "${TEST_LABEL_JSONS[@]}" \
  --test_dataset_names "${TEST_DATASET_NAMES[@]}" \
  --epoch "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --img_size "${IMG_SIZE}" \
  --cuda_device "${CUDA_DEVICE}" \
  --dataset_name "${DATASET_NAME}"

