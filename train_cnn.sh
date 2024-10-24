#!/bin/bash

set -x

bash scripts/train.sh densenet161 1 1000
bash scripts/train.sh densenet161 1 2000
bash scripts/train.sh densenet121 1 1000
bash scripts/train.sh densenet121 1 2000
bash scripts/train.sh resnet34 1 1000
bash scripts/train.sh resnet34 1 2000