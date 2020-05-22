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
PYTHONPATH=./ python aut_experiments/200426_splits_calc_feats.py \
    --data-path ../../preproc_data/autists \
    --out-path ../own_data/200426_autists_features_v2_thr30 \
    --debug

"""

import argparse
from os.path import join, exists
from os import mkdir

import pandas as pd

from pipeline.data_preparation import FeatureBuilder, get_train_df


feature_files = {
    'coh_alpha.csv': ['coh-alpha'],
    'coh_beta.csv': ['coh-beta'],
    'coh_theta.csv': ['coh-theta'],
    'env_alpha.csv': ['env-alpha'],
    'env_beta.csv': ['env-beta'],
    'env_theta.csv': ['env-theta'],
    'bands.csv': ['bands'],
    'set_1.csv': ['bands', 'coh-alpha', 'coh-beta', 'env-alpha', 'env-beta'],
    'set_2.csv': ['bands', 'coh-alpha', 'coh-beta', 'coh-theta', 'env-alpha', 'env-beta', 'env-theta'],
}


def main():
    # actual min - 13
    # thr = 13
    thr = 30
    # thr = 60

    data_path, out_path, debug = parse_args()
    path_df = pd.read_csv(join(data_path, 'path_file.csv'))
    path_df = path_df[path_df['seconds'] >= thr]

    if not exists(out_path):
        mkdir(out_path)

    for split_type in ['full', 'part_1', 'part_2']:
        # for split_type in ['part_1', 'part_2']:
        if not exists(join(out_path, split_type)):
            mkdir(join(out_path, split_type))

        feature_builder_dict = {}
        for out_fn, method_names in feature_files.items():
            feature_builder_dict[out_fn] = FeatureBuilder(method_names=method_names)
        cnt = 0
        for i, row in path_df.iterrows():
            df = pd.read_csv(join(data_path, row['fn']), index_col='time')
            df = df.iloc[:thr*125]

            df = get_split(df, split_type)
            print(row['fn'])
            del df['cz']
            for out_fn, builder in feature_builder_dict.items():
                builder.add(df, row['fn'])
            cnt += 1
            if debug and cnt > 3:
                break

        for out_fn, builder in feature_builder_dict.items():
            features_df = builder.get_df()
            train_df = get_train_df(features_df, path_df)
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
