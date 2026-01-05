#!/bin/bash
accelerate launch \
  --config_file configs/accelerate.yaml \
  -m src.training.train
