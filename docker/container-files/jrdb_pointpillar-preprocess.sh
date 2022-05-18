#!/bin/bash
source /workspace/ros_entrypoint.sh
cd /workspace/Open3D-ML
pip3 install -r requirements-torch-cuda.txt
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/Open3D-ML

./ml3d/datasets/utils/jrdb_preprocessing.py /workspace/jrdb/train_dataset_with_activity/pointclouds /workspace/Open3D-ML/ml3d/configs/jrdb_configs
./ml3d/datasets/utils/jrdb_preprocessing.py /workspace/jrdb/test_dataset_without_labels/pointclouds /workspace/Open3D-ML/ml3d/configs/jrdb_configs nolabels
