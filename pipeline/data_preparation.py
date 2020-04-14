from os import listdir
from os.path import join

import pandas as pd

from .features import calc_features_dict


"""
Требования к пайплайну

Есть папка с предобработанными файлами, с path_file.csv с таргетами
Нужно подготовить csv файлы с признаками - отдельно каждыми
Нужно уметь мерджить различные признаки
Нужно уметь сделать тест различных моделей на различных признаках

Уйти от длинных CLI и неудобных sh скриптов - перейти к питоновским скриптам 
"""


def get_train_df(features_df, path_df):
    assert 'target' in path_df == True, 'No target in path_df'
    res_df = features_df.merge(path_df[['fn', 'target']], on='fn')
    return res_df


class FeatureBuilder:
    def __init__(self, method_names):
        self.rows = []
        self.method_names = method_names

    def add(self, df, fn):
        d = calc_features_dict(df, self.method_names)
        d['fn'] = fn
        self.rows.append(d)

    def get_df(self):
        features_df = pd.DataFrame(self.rows)
        return features_df


def merge_dfs(dfs=None, in_paths=None, out_path=None):
    """
    Args:
        dfs: list
        in_paths: list
        out_path: str

    Returns:
         res_df: pd.DataFrame, optional

    Takes dfs or in_paths as input, not both

    E.g. in_paths = ['features/coh_alpha.csv', 'features/env_alpha.cvs']

    returns DataFrame if out_path is not provided
    returns None and saves DataFrame to csv if out_path is provided


    """
    if not (dfs is None ^ in_paths is None):
        raise ValueError('You should provide either dfs or fns')

    if in_paths is not None:
        dfs = []
        for in_path in in_paths:
            dfs.append(pd.read_csv(in_path))

    res_df = pd.DataFrame()

    # data check
    for df in dfs:
        assert 'target' in df.columns, 'Error: no target column'
        assert 'fn' in df.columns, 'Error: no fn column'
    for df_1, df_2 in zip(dfs[:-1], dfs[1:]):
        assert set(df_1['fn'].tolist()) == set(df_2['fn'].tolist()), 'Error: fns are not equal'

    for i, df in enumerate(dfs):
        df = df.copy()
        if i == 0:
            res_df = df
        else:
            del df['target']
            res_df = res_df.merge(df, on='fn')

    if out_path:
        res_df.to_csv(out_path, index=False)
    else:
        return res_df



