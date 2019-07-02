import argparse
import os
from os.path import join, exists
import re

import pandas as pd

from mne.io import read_raw_edf

from paths import get_path_df


def main():
    data_path, out_path, skip_existing = parse_args()

    path_df = get_path_df(data_path)

    create_if_necessary(out_path)

    for i, row in path_df.iterrows():

        path_to_save = join(out_path, row['fn'])
        if skip_existing and exists(path_to_save):
            continue

        print(row['fn'])

        # Currently we can not work with these records
        if row['fn'][:-4] in ['17m_As_fon', '15m_Lp_fon', '17m_Au_fon', 'dev11_og']:
            print('Unusual channel names', row['fn'])
            continue
        if row['fn'][:-4] == 'KOMLEVA_ORG_ASD_F':
            print('Cant read KOMLEVA_ORG_ASD_F')
            continue

        raw_file = read_raw_edf(row['full_path'], verbose=False)
        df = raw_file.to_data_frame()
        df = change_ref_to_a1(df, row['fn'])

        # reduce sfreq to 125Hz
        df = df.iloc[::2, :]
        df.to_csv(path_to_save)

    del path_df['full_path']
    path_df.to_csv(join(out_path, 'path_file.csv'), index=False)


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


def create_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_epoches(path_file_path, out_path, fn='', debug=False):

    create_if_necessary(out_path)

    path_df = pd.read_csv(path_file_path)

    for i, row in path_df.iterrows():
        path_to_save = join(out_path, row['fn'] + '.csv')
        print(row['fn'])
        if row['fn'] in ['17m_As_fon', '15m_Lp_fon', '17m_Au_fon']:
            print('Error in fon, skip')
            continue
        if row['fn'] == 'KOMLEVA_ORG_ASD_F':
            print('Cant read KOMLEVA_ORG_ASD_F')
            continue
        if row['fn'] == 'dev11_og':
            print('dev11_og - something wrong in column names')
            continue
        raw_file = read_raw_edf(row['full_path'], verbose=False)
        df = raw_file.to_data_frame()
        df = change_ref_to_a1(df, row['fn'])
        # df = change_reference(df, 'cz')
        # reduce sfreq to 125Hz
        df = df.iloc[::2, :]
        df.to_csv(path_to_save)


def find_channels(s):
    return re.findall(r'([a-z0-9]+)-([a-z0-9]+)', s.lower())[0]


def change_ref_to_a1(df, fn):
    channels = {}
    channels_rev = {}
    for col in df.columns:
        if 'eeg' not in col.lower():
            continue
        ch1, ch2 = find_channels(col)
        channels[col] = (ch1, ch2)
        channels_rev[(ch1, ch2)] = col

    if ('a1', 'a2') in channels_rev:
        col = channels_rev[('a1', 'a2')]
        a1_a2 = df[col]

    elif ('a2', 'a1') in channels_rev:
        col = channels_rev[('a2', 'a1')]
        a1_a2 = - df[col]
    else:
        print('Error in ' + fn)
        return None

    new_df = df.iloc[:, :0].copy()

    for col in df.columns:
        if 'eeg' not in col.lower():
            continue
        ch1, ch2 = channels[col]
        if ch1 == 'a1' or ch1 == 'a2':
            continue
        if ch2 == 'a1':
            new_df[ch1] = df[col]
        elif ch2 == 'a2':
            sig = df[col] - a1_a2
            new_df[ch1] = sig

        new_df['a2'] = - a1_a2
        new_df['a1'] = 0

    return new_df


if __name__ == '__main__':
    main()
