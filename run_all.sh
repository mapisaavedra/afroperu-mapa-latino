#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src"
python prepare_data.py
python train.py
python peru_analysis.py
