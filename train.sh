#!/bin/bash

set -x

bash scripts/train.sh densenet161 1
bash scripts/train.sh resnet34 1