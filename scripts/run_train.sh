#!/bin/bash
LOG_DIR="runs/logs"
mkdir -p $LOG_DIR
# --cfg configs/yolo11n.yaml
python src/train.py | tee $LOG_DIR/training_log_$(date +%Y%m%d_%H%M%S).txt
