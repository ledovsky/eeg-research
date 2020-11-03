from numpy import random
import pandas as pd
import numpy as np
from scipy.sparse.construct import rand

from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from .models import PredictionsResult, train_test, select_features, get_x_y


def train_test_val(df, features, model, test_size=15, verbose=False, random_state=42):
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, stratify=df['target'], random_state=random_state)
    new_features, _, _ = select_features(df.loc[train_idx], features, model, verbose=verbose, n_repeats=1)
    X_1, y_1 = get_x_y(df.loc[train_idx], new_features)
    X_2, y_2 = get_x_y(df.loc[test_idx], new_features)
    return new_features, train_test(X_1, y_1, X_2, y_2, model)


def nested_cross_val(df, features, model, n_splits=10, n_repeats=10, verbose=False, random_state=42):
    np.random.seed(random_state)
    y_preds = np.empty((df.shape[0]))
    cv = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in cv.split(df):
        new_features, _, _, = select_features(df.loc[train_idx], features, model, n_repeats=n_repeats, verbose=verbose)
        X_train, y_train = get_x_y(df.loc[train_idx], new_features)
        X_test, _ = get_x_y(df.loc[test_idx], new_features)
        model.fit(X_train, y_train)
        y_preds[test_idx] = model.predict_proba(X_test)[:, 1]
    return PredictionsResult(df['target'], y_preds)


def repeated_train_test(df, features, model, test_size=15, verbose=False, n_repeats=10,
                        random_state=42):
    np.random.seed(random_state)
    scores = []
    best_features = []
    random_states = np.random.randint(100000, size=n_repeats)
    for i in range(n_repeats):
        feats, score = train_test_val(df, features, model, test_size, verbose, random_states[i])
        best_features.append(feats)
        scores.append(score)
    return best_features, scores


def multi_segment_train_test(df, features, model, test_size=15, verbose=False, random_state=42):
    idx = df['fn'].drop_duplicates()
    fn_df = df.set_index('fn')
    targets = fn_df[~fn_df.index.duplicated('first')]['target'][idx] # Get target for each id
    train_ids, test_ids = train_test_split(idx, test_size=test_size, stratify=targets, random_state=random_state)
    new_features, _, _ = select_features(fn_df.loc[train_ids], features, model, verbose=verbose, n_repeats=1)
    X_1, y_1 = get_x_y(fn_df.loc[train_ids], new_features)
    X_2, y_2 = get_x_y(fn_df.loc[test_ids], new_features)
    return new_features, train_test(X_1, y_1, X_2, y_2, model)