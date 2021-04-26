"""
Example usage:
PYTHONPATH=./ python asd_preproc/main.py \
    --data-path ../../raw_data/2104_asd \
    --out-path ../../preproc_data/asd
"""

import argparse
import os
from os.path import join
import pandas as pd

from mne.io import read_raw_edf

from pipeline.utils import create_if_necessary
from pipeline.data_preparation import add_montage
from asd_preproc.paths import get_path_df


chs_to_use = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
               'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']

def main():
    data_path, out_path, skip_existing = parse_args()

    path_df = get_path_df(data_path, verbose=False)
    create_if_necessary(out_path)

    rows = []

    l_freq = 1
    h_freq = 40
    new_sfreq = 125

    for i, row in path_df.iterrows():

        print(row['fn'])
        raw = read_raw_edf(row['full_path'], verbose=False, preload=True)

        ch_map = {ch: find_channel(ch, chs_to_find=chs_to_use) for ch in raw.ch_names}
        raw.rename_channels(ch_map)
        raw.pick_channels(chs_to_use)
        raw = add_montage(raw, ch_map=None)

        # do filtering and resample due to lack of events data
        # in case of events it's recommended to get epochs data before
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.resample(new_sfreq)

        new_fn = row['fn'].replace('.edf', '.raw.fif')
        raw.save(join(out_path, new_fn), overwrite=True)
        rows.append({
            'fn': new_fn,
            'target': row['target'],
            'dataset_name': row['dataset_name'],
            'sfreq': new_sfreq,
            'age': row['age'],
            'seconds': row['seconds'],
        })

    pd.DataFrame(rows).to_csv(join(out_path, 'path_file.csv'), index=False)


def find_channel(ch, chs_to_find):

    for ch_to_find in chs_to_find:
        if ch_to_find.lower() in ch.lower():
            return ch_to_find
    return ch


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--out-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--skip-existing', action='store_true',
                        help='')

    args = parser.parse_args()

    return args.data_path, args.out_path, args.skip_existing


if __name__ == '__main__':
    main()
