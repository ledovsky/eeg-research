import os
from tqdm.auto import tqdm

def get_x_y(df, features):
    X = df[features].fillna(0).values
    y = df['target'].values
    return X, y


def get_x(df, features):
    X = df[features].fillna(0).values
    return X


def get_tqdm_iter(iterable, p_bar, **tqdm_kwargs):
    if p_bar > 0:
        return tqdm(iterable, **tqdm_kwargs)
    return iterable


def create_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)
