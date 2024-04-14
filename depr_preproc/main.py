import argparse
import os
from os.path import join, exists
import re

import pandas as pd

import mne
from mne.io import read_raw_egi

from paths import get_path_df
from utils import get_events, get_epoch, get_epoches
from electrodes import electrode_mapping, electrodes


def main():
    data_path, out_path, only_10_20, fn = parse_args()

    path_df = get_path_df(data_path)

    create_if_necessary(out_path)
    path_to_epoch_1 = join(out_path, 'epoch_1')
    path_to_epoch_2 = join(out_path, 'epoch_2')
    path_to_epoch_3 = join(out_path, 'epoch_3')
    create_if_necessary(path_to_epoch_1)
    create_if_necessary(path_to_epoch_2)
    create_if_necessary(path_to_epoch_3)

    if fn:
        row = path_df[path_df['fn'] == fn].iloc[0]
        if not len(row):
            assert ValueError('fn is not exist')
        process_row(row, path_to_epoch_1, path_to_epoch_2, path_to_epoch_3, only_10_20=only_10_20)

    else:
        for i, row in path_df.iterrows():
            print(row['fn'])
            try:
                process_row(row, path_to_epoch_1, path_to_epoch_2, path_to_epoch_3, only_10_20=only_10_20)
            except AssertionError:
                print('Error in ' + row['full_path'])
                continue

    del path_df['full_path']
    path_df.to_csv(join(path_to_epoch_1, 'path_file.csv'), index=False)
    path_df.to_csv(join(path_to_epoch_2, 'path_file.csv'), index=False)
    path_df.to_csv(join(path_to_epoch_3, 'path_file.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', action='store', type=str, required=True, help='')

    parser.add_argument('--out-path', action='store', type=str, required=True, help='')

    parser.add_argument('--only-10-20', action='store_true', help='')

    parser.add_argument('--fn', action='store', type=str, default='', help='')

    args = parser.parse_args()

    return args.data_path, args.out_path, args.only_10_20, args.fn


def create_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_row(row, path_to_epoch_1, path_to_epoch_2, path_to_epoch_3, only_10_20=True):
    sample = read_raw_egi(row['full_path'], verbose=0)
    df = sample.to_data_frame()

    n_epoches = 3
    epoch_types = None
    if row['fn'][:-4] == 'Smirnova rest 24-03':
        n_epoches=4
        epoch_types = [0, 1, 0, 1]
    elif row['fn'][:-4] == 'Saradjev rest 16-11':
        n_epoches=5
        epoch_types = [0, 1, 0, 0, 0]

    events = get_events(df, n_epoches=n_epoches, epoch_types=epoch_types)

    picks = mne.pick_types(sample.info, meg=False, eeg=True)

    df = get_epoches(df, picks, events)

    columns_to_use = [col for col in df.columns if col[0] == 'E']
    df = df.loc[:, columns_to_use]

    def replace_if_possible(col):
        if col in electrode_mapping:
            return electrode_mapping[col]
        else:
            return col

    df.columns = [replace_if_possible(col) for col in df.columns]

    df['cz'] = 0

    if only_10_20:
        df = df[electrodes]

    # save epoch 1
    df_epoch = get_epoch(df, 0)
    df_epoch = df_epoch.iloc[:30000][::4]
    df_epoch.to_csv(join(path_to_epoch_1, row['fn']))

    # save epoch 2
    idx = 1
    if row['fn'][:-4] == 'Smirnova rest 24-03':
        idx = 3
    df_epoch = get_epoch(df, idx)
    df_epoch = df_epoch.iloc[:30000][::4]
    df_epoch.to_csv(join(path_to_epoch_2, row['fn']))

    # save epoch 3
    idx = 2
    if row['fn'][:-4] == 'Saradjev rest 16-11':
        idx = 4
    df_epoch = get_epoch(df, idx)
    df_epoch = df_epoch.iloc[:30000][::4]
    df_epoch.to_csv(join(path_to_epoch_3, row['fn']))


if __name__ == '__main__':
    main()
