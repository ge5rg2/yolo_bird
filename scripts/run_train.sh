#!/bin/bash
LOG_DIR="runs/logs"
mkdir -p $LOG_DIR

python src/train.py --cfg configs/yolo11n.yaml | tee $LOG_DIR/training_log_$(date +%Y%m%d_%H%M%S).txt
