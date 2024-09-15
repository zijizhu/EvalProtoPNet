#!/bin/bash

set -x

python eval_nmi_ari.py --base_architecture dinov2_vitb_exp --resume output_cosine/CUB2011/dinov2_vitb_exp/1028-1e-4-adam-12-train/checkpoints/save_model.pth
