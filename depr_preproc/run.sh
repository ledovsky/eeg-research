#!/usr/bin/env bash

# Example run

data_path="/Users/zubrikhina/Documents/datasets/depr_evm/"
out_path="/Users/zubrikhina/Documents/datasets/depr_evm/preproc_data/depr_2"

python main.py --data-path=$data_path --out-path=$out_path --only-10-20
