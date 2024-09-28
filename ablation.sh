#!/bin/bash

set -x

bash train_vit.sh dinov2_vitb_exp 1 False False False xe_only
bash train_vit.sh dinov2_vitb_exp 1 True False False clst_sep
bash train_vit.sh dinov2_vitb_exp 1 True True False clst_sep_orth
bash train_vit.sh dinov2_vitb_exp 1 True True True all

bash train_vit.sh dinov2_vits_exp 1 False False False xe_only
bash train_vit.sh dinov2_vits_exp 1 True False False clst_sep
bash train_vit.sh dinov2_vits_exp 1 True True False clst_sep_orth
bash train_vit.sh dinov2_vits_exp 1 True True True all
