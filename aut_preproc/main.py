"""
Example usage:
python aut_preproc/main.py \
    --data-path ../../raw_data/autists \
    --out-path ../../preproc_data/autists
"""

import argparse
import os
from os.path import join

import pandas as pd

from mne.io import read_raw_edf

from paths import get_path_df


def main():
    data_path, out_path, skip_existing = parse_args()

    path_df = get_path_df(data_path)

    create_if_necessary(out_path)

    results = []
    to_exclude = []

    for i, row in path_df.iterrows():

        path_to_save = join(out_path, row['fn'])

        raw_file = read_raw_edf(row['full_path'], verbose=False, preload=True)
        df = raw_file.to_data_frame()

        try:
            df, res = process_data(df)
            results.append(['OK', row['full_path'], row['fn'], res])
        except Exception as e:
            results.append(['FAILURE', row['full_path'], row['fn'], str(type(e)) + ': ' + str(e)])
            to_exclude.append(i)
            continue

        df = change_reference(df, 'cz')
        # reduce sfreq to 125Hz
        df = df.iloc[::2, :]
        df.to_csv(path_to_save)

    del path_df['full_path']
    path_df = path_df.loc[~path_df.index.isin(to_exclude)]
    path_df['sfreq'] = 125
    path_df.to_csv(join(out_path, 'path_file.csv'), index=False)

    df_results = pd.DataFrame(results, columns=['status', 'full_path', 'fn', 'comment'])
    df_results.to_csv(join(out_path, 'processing_log.csv'), index=False)


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


def change_reference(df, new_ref):
    df_old = df.copy()
    df = df.copy()
    for col in df.columns:
        df[col] = df[col] - df_old[new_ref]

    return df


def process_data(df):
    """
    Логика

    - Приводим к нижнему регистру
    - Убираем префикс eeg
    - Ищем каналы без префиксов eeg. Если нашли 21 - все ок
    - Ищем канал a1-a2 или a2-a1. Если находим, создаем a1 и a2
    - Если есть каналы a2-a1 и а1, либо a1-a2 и a2 обрабатываем это отдельно
    - Ищем остальные каналы с префиксом eeg и приводим их к общему референсу

    Args:
        df: pd.DataFrame

    Returns:
        out_df: pd.DataFrame

    """

    df.set_index('time', inplace=True)

    desired_columns = ['fp1', 'fp2', 'f7', 'f3', 'fz', 'f4', 'f8', 't3', 'c3', 'cz', 'c4',
                       't4', 't5', 'p3', 'pz', 'p4', 't6', 'o1', 'o2']

    df.columns = [col.lower() for col in df.columns]
    df.columns = [col.replace('eeg ', '') for col in df.columns]

    a1_ref_columns = [(col, col + '-a1') for col in desired_columns]
    a2_ref_columns = [(col, col + '-a2') for col in desired_columns]

    not_found = []
    for col in desired_columns:
        if col not in df.columns:
            not_found.append(col)

    if len(not_found) == 0:
        return df[desired_columns], 'all channels found'

    a1_a2_str = ''
    if 'a1-a2' in df.columns:
        if 'a2' not in df.columns:
            df['a2'] = 0
        else:
            a1_a2_str = '; a1-a2 + a2 present'
        df['a1'] = df['a1-a2'] + df['a2']
    elif 'a2-a1' in df.columns:
        if 'a1' not in df.columns:
            df['a1'] = 0
        else:
            a1_a2_str = '; a2-a1 + a1 present'
        df['a2'] = df['a2-a1'] + df['a1']
    else:
        raise Exception('no ref found or not enough columns' + '; ' + '|'.join(not_found))

    for col, ref_col in a1_ref_columns:
        if ref_col in df.columns:
            df[col] = df[ref_col] + df['a1']
    for col, ref_col in a2_ref_columns:
        if ref_col in df.columns:
            df[col] = df[ref_col] + df['a2']

    not_found = []
    for col in desired_columns:
        if col not in df.columns:
            not_found.append(col)

    if len(not_found) == 0:
        return df[desired_columns], 'all channels found; A1 and A2 references changed' + a1_a2_str

    else:
        raise Exception('not enough columns; A1 and A2 references changed' + a1_a2_str +
                        '; ' + '|'.join(not_found))


if __name__ == '__main__':
    main()
