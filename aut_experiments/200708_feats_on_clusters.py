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
PYTHONPATH=./ python aut_experiments/200708_feats_on_clusters.py \
    --data-path ../../preproc_data/autists \
    --out-path ../own_data/200708_aut_clust_v1 \
    --cluster-type 0 \
    --debug

"""

import argparse
from os.path import join, exists
from os import mkdir

import pandas as pd

from pipeline.data_preparation import FeatureBuilder, get_train_df

"""
(F3, F7, T3, C3);     (F4, F8, T4, C4);   (T5, P3, O1);  (T6, P4, O2);  (Fz, Cz, Pz)

Второй вариант:

(F3, Fz, F4) (F7, T3), (F8, T4), (C3, Cz, C4) (P3, Pz, P4), (T5, O1), (T6, O2)

Третий вариант:

(F3, F7, T3), (F4, F8, T4), (C3, Cz, C4, Pz), (T5, O1, Р3), (T6, O2, P4)

"""

clusters_0 = [
    ('f3', 'f7', 't3', 'c3'),
    ('f4', 'f8', 't4', 'c4'),
    ('t5', 'p3', 'o1'),
    ('t6', 'p4', 'o2'),
    ('fz', 'cz', 'pz')
]

clusters_1 = [
    ('f3', 'fz', 'f4'),
    ('f7', 't3'),
    ('c3', 'cz', 'c4'),
    ('p3', 'pz', 'p4'),
    ('t5', 'o1'),
    ('t6', 'o2')
]

clusters_2 = [
    ('f3', 'f7', 't3'),
    ('f4', 'f8', 't4'),
    ('c3', 'cz', 'c4', 'pz'),
    ('t5', 'o1', 'p3'),
    ('t6', 'o2', 'p4')
]

def sum_by_clusters(df, clusters):
    new_df = df[[]].copy()
    for i, clust in enumerate(clusters):
        new_ch_name = 'clust_' + str(i)
        new_df[new_ch_name] = df[list(clust)].sum(axis=1)
    return new_df

feature_files = {
    'coh_alpha.csv': ['coh-alpha'],
    'coh_beta.csv': ['coh-beta'],
    'coh_theta.csv': ['coh-theta'],
    'env_alpha.csv': ['env-alpha'],
    'env_beta.csv': ['env-beta'],
    'env_theta.csv': ['env-theta'],
    'bands.csv': ['bands'],
    'set_1.csv': ['coh-alpha', 'coh-beta', 'env-alpha', 'env-beta'],
    'set_2.csv': ['coh-alpha', 'coh-beta', 'coh-theta', 'env-alpha', 'env-beta', 'env-theta'],
}


def main():
    # actual min - 13
    # thr = 13
    thr = 30
    # thr = 60

    data_path, out_path, cluster_type, debug = parse_args()

    if cluster_type == 0:
        clusters = clusters_0
    elif cluster_type == 1:
        clusters = clusters_1
    elif cluster_type == 2:
        clusters = clusters_2
    else:
        raise ValueError('cluster type should be 0, 1 or 2')

    path_df = pd.read_csv(join(data_path, 'path_file.csv'))
    path_df = path_df[path_df['seconds'] >= thr]

    if not exists(out_path):
        mkdir(out_path)

    for split_type in ['full', 'part_1', 'part_2']:
        if not exists(join(out_path, split_type)):
            mkdir(join(out_path, split_type))

        feature_builder_dict = {}
        for out_fn, method_names in feature_files.items():
            feature_builder_dict[out_fn] = FeatureBuilder(method_names=method_names)
        cnt = 0
        for i, row in path_df.iterrows():
            df = pd.read_csv(join(data_path, row['fn']), index_col='time')
            df = df.iloc[:thr*125]
            df = sum_by_clusters(df, clusters)
            df = get_split(df, split_type)
            print(row['fn'])
            # del df['cz']
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

    parser.add_argument('--cluster-type', action='store', type=int, required=True,
                        help='')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    return args.data_path, args.out_path, args.cluster_type, args.debug


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
