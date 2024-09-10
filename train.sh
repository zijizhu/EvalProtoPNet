#!/bin/bash

set -x

bash scripts/train.sh dinov2_vitb_exp 1
bash scripts/train.sh dinov2_vits_exp 1
