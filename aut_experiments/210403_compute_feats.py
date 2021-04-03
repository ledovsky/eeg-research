"""
Отличие от предыдущей части.

Предварительно производим фиильтрацию записей, содержащих не менее thr секунд
Из всех файлов берем первые thr секунд. Таким образом длина всех записей будет одинаковая

Пробуем версии со всеми файлами (thr = 13), и с другими thr
Кажется, что оптимально взять 30, но может быть и 13 будет нормально


Логика сплитов

Считаем признаки по всему файл

Делим каждый файл на 2 части, считаем признаки по ним

Раскладываем по папкам

own_data/autists_splits
- full
- part_1
- part_2


Example how to run:
PYTHONPATH=./ python aut_experiments/210403_compute_feats.py \
    --data-path ../../preproc_data/autists \
    --out-path ../own_data/210403_aut_bands_and_env_var \
    --debug
"""

import argparse
from os.path import join, exists
from os import mkdir
from functools import partial

import pandas as pd

from pipeline.data_preparation import FeatureBuilder, get_train_df
from pipeline.features import get_bands_feats, get_envelope_std




def main():

    thr = 30
    sfreq = 125
    out_fn = 'features.csv'

    data_path, out_path, debug = parse_args()
    path_df = pd.read_csv(join(data_path, 'path_file.csv'))
    path_df = path_df[path_df['seconds'] >= thr]

    if not exists(out_path):
        mkdir(out_path)

    for split_type in ['full', 'part_1', 'part_2']:
        # for split_type in ['part_1', 'part_2']:
        if not exists(join(out_path, split_type)):
            mkdir(join(out_path, split_type))

        feature_builder = FeatureBuilder(methods=[
            partial(get_bands_feats, bands=['4_6', '6_8', '8_10', '10_12', 'beta'], nperseg=512),
            partial(get_envelope_std, band='alpha')
        ])

        cnt = 0
        for i, row in path_df.iterrows():
            df = pd.read_csv(join(data_path, row['fn']), index_col='time')

            df = df.iloc[:thr*sfreq]

            df = get_split(df, split_type)

            # average referencing
            mean = df.mean(axis=1)
            for col in df.columns:
                df.loc[:, col] = df.loc[:, col] - mean

            print(row['fn'])
            del df['cz']
            feature_builder.process_sample(df, row['fn'])
            cnt += 1
            if debug and cnt > 3:
                break

        features_df = feature_builder.get_df()
        train_df = get_train_df(features_df, path_df, columns_to_add=['target', 'age'])
        train_df.to_csv(join(out_path, split_type, out_fn), index=False)

        if debug:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--data-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--out-path', action='store', type=str, required=True,
                        help='')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args.data_path, args.out_path, args.debug


def get_split(df, split_type):
    n = df.shape[0]
    if split_type == 'full':
        return df
    elif split_type == 'part_1':
        return df.iloc[:int(n / 2)]
    elif split_type == 'part_2':
        return df.iloc[int(n / 2):]


if __name__ == '__main__':
    main()
