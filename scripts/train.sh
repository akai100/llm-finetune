#!/bin/bash
accelerate launch \
  --config_file configs/accelerate.yaml \
  src/training/train.py
