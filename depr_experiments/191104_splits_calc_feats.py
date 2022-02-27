import sys
from os.path import exists, join
from shutil import rmtree
from os import mkdir

sys.path.append('../pipeline')

from features import get_feature_build_func


# data_path_1 = '../../../preproc_data/depr/epoch_1'
# data_path_2 = '../../../preproc_data/depr/epoch_3'
# data_paths = [
#     data_path_1,
#     data_path_2
# ]
# out_path = '../../own_data/191104_depr_splits_features'

data_paths = ['../data/asd/csv']

out_path = '../data/asd/features'

# split length, epoch_idx, start_sec, end_sec

splits_60sec = [
    ('60s', 0, 0, 60),
    ('60s', 1, 0, 60),
]


splits_30sec = [
    ('30s', 0, 0, 30),
    ('30s', 0, 30, 60),
    ('30s', 1, 0, 30),
    ('30s', 1, 30, 60),
]


splits_20sec = [
    ('20s', 0, 0, 20),
    ('20s', 0, 20, 40),
    ('20s', 0, 40, 60),
    ('20s', 1, 0, 20),
    ('20s', 1, 20, 40),
    ('20s', 1, 40, 60),
]


splits_10sec = [
    ('10s', 0, 0, 10),
    ('10s', 0, 10, 20),
    ('10s', 0, 20, 30),
    ('10s', 0, 30, 40),
    ('10s', 0, 40, 50),
    ('10s', 0, 50, 60),
    ('10s', 1, 0, 10),
    ('10s', 1, 10, 20),
    ('10s', 1, 20, 30),
    ('10s', 1, 30, 40),
    ('10s', 1, 40, 50),
    ('10s', 1, 50, 60),
]

feature_methods = [
    'coh-alpha',
    'coh-beta',
    'coh-theta',
    'env-alpha',
    'env-beta',
    'env-theta',
    'bands',
    'psi'
]


def get_df_filter_func(start_sec, end_sec):
    start_idx = start_sec * 125
    end_idx = end_sec * 125
    def wrapped(df):
        return df.iloc[start_idx:end_idx, :]
    return wrapped


def get_split_path(split, path):
    dir_name = '_'.join([split[0], str(split[1]), str(split[2]), str(split[3])])
    return join(path, dir_name)


def calc_features_by_splits():
    # splits = splits_10sec + splits_20sec + splits_30sec + splits_60sec
    splits = [('all', 0, 0, 20), ('val', 0, 20, 40)]
    for split in splits:
        df_filter_func = get_df_filter_func(split[2], split[3])
        for method in feature_methods:

            split_out_path = get_split_path(split, out_path)
            split_data_path = data_paths[split[1]]
            if not exists(split_out_path):
                mkdir(split_out_path)

            extract_features = get_feature_build_func(method, df_filter_func=df_filter_func)
            extract_features(split_data_path, split_out_path)


if __name__ == '__main__':
    if exists(out_path):
        rmtree(out_path)
    mkdir(out_path)

    calc_features_by_splits()


