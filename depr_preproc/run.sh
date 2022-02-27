#!/usr/bin/env bash

# Example run

data_path="../../../raw_data"
out_path="../../../preproc_data/depr_2"

python main.py --data-path=$data_path --out-path=$out_path --only-10-20
