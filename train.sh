#!/bin/bash

set -x

bash train_cars.sh dinov2_vitb_exp 1 980
bash train_cars.sh dinov2_vits_exp 1 980
bash train_cars.sh dinov2_vitb_exp 1 588
bash train_cars.sh dinov2_vits_exp 1 588

# bash scripts/train.sh dinov2_vitb_exp 1 600
# bash scripts/train.sh dinov2_vitb_exp 1 1000
# bash scripts/train.sh dinov2_vitb_exp 1 2000

# bash scripts/train.sh dino_vitb8 1 2000
# bash scripts/train.sh dino_vits8 1 2000

# bash scripts/train.sh dinov2_vits_exp 1 600
# bash scripts/train.sh dinov2_vits_exp 1 1000
# bash scripts/train.sh dinov2_vits_exp 1 2000

# bash scripts/train.sh dino_vitb16 1 2000
# bash scripts/train.sh dino_vitb16 1 1000
# bash scripts/train.sh dino_vitb16 1 600

# bash scripts/train.sh dino_vitb8 1 600
# bash scripts/train.sh dino_vitb8 1 1000
# bash scripts/train.sh dino_vitb8 1 2000
# bash scripts/train.sh dino_vits8 1 600
# bash scripts/train.sh dino_vits8 1 1000
# bash scripts/train.sh dino_vits8 1 2000
