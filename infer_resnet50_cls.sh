DATASET_NAME="TN3K"
# 对其他良恶性数据集做推理
python infer_resnet50_cls.py \
  --image_dir "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/${DATASET_NAME}/test/images" \
  --label_json "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/${DATASET_NAME}/test/${DATASET_NAME}_test_label.json" \
  --task_key "malignancy" \
  --checkpoint_path "/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/resnet50/checkpoints_resnet50/resnet50_thyroid_cls_resnet50_epoch_50.pth" \
  --output_dir "./infer/${DATASET_NAME}"