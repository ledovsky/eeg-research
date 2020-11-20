import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

from .models import PredictionsResult, train_test, select_features, get_x, get_x_y, get_tqdm_iter


def train_test_val(df, features, model, test_size=15, metric='roc-auc', random_state=42, verbose=False, p_bar=0):
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, stratify=df['target'], random_state=random_state)

    if verbose:
        print('Data split.')
        print('train indices:', np.sort(train_idx.values), 
              'train indices:', np.sort(test_idx.values), sep='\n')
        print()

    new_features, _, _ = select_features(df.loc[train_idx], features, model, metric, n_repeats=1, verbose=verbose, p_bar=p_bar)
    X_1, y_1 = get_x_y(df.loc[train_idx], new_features)
    X_2, y_2 = get_x_y(df.loc[test_idx], new_features)
    return new_features, train_test(X_1, y_1, X_2, y_2, model)


def nested_cross_val(df, features, model, n_splits=10, n_repeats=10, metric='roc-auc', random_state=42, verbose=False, p_bar=1):
    np.random.seed(random_state)
    y_preds = np.empty((df.shape[0]))
    cv = KFold(n_splits=n_splits, shuffle=True)
    for train_idx, test_idx in get_tqdm_iter(cv.split(df), p_bar, total=n_splits):

        if verbose:
            print('Data split.')
            print('train indices:', np.sort(train_idx), 
                'train indices:', np.sort(test_idx), sep='\n')
            print()

        new_features, _, _, = select_features(df.loc[train_idx], features, model, metric, n_repeats=n_repeats, verbose=verbose, p_bar=p_bar-1)
        X_train, y_train = get_x_y(df.loc[train_idx], new_features)
        X_test = get_x(df.loc[test_idx], new_features)
        model.fit(X_train, y_train)
        y_preds[test_idx] = model.predict_proba(X_test)[:, 1]
    return PredictionsResult(df['target'], y_preds)


def get_repeated_scores(method, *method_args, method_kwargs={}, n_repeats=10, verbose=False, p_bar=1, random_state=42):
    np.random.seed(random_state)
    scores = []
    random_states = np.random.randint(100000, size=n_repeats)
    for i in get_tqdm_iter(range(n_repeats), p_bar):
        if verbose:
            print(f'Iteration {i}.')
        _, score = method(*method_args, random_state=random_states[i], verbose=verbose, p_bar=p_bar-1, **method_kwargs)
        scores.append(score)
    return scores


def multi_segment_train_test(df, features, model, test_size=15, metric='roc-auc', random_state=42, verbose=False, p_bar=0):
    idx = df['fn'].drop_duplicates()
    fn_df = df.set_index('fn')
    targets = fn_df[~fn_df.index.duplicated('first')]['target'][idx] # Get target for each id
    train_ids, test_ids = train_test_split(idx, test_size=test_size, stratify=targets, random_state=random_state)
    new_features, _, _ = select_features(fn_df.loc[train_ids], features, model, metric, n_repeats=1, verbose=verbose, p_bar=p_bar)
    X_1, y_1 = get_x_y(fn_df.loc[train_ids], new_features)
    X_2 = get_x(fn_df.loc[test_ids], new_features)

    model.fit(X_1, y_1)
    predicted = pd.Series(model.predict_proba(X_2)[:, 1], fn_df.loc[test_ids].index, name='pred').groupby('fn').mean()
    results = pd.concat([targets.loc[test_ids], predicted], axis=1)
    return new_features, PredictionsResult.from_df(results)
