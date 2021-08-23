import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

from .models import PredictionsResult, train_test, select_features, CVPredictionsResult
from .utils import get_x, get_x_y, get_tqdm_iter


class FeatureSelectionModel(object):
    def __init__(self, df, features, model, verbose=False, p_bar=0, random_state=42):
        self.df = df
        self.features = features
        self.model = model

        self.verbose = verbose
        self.p_bar = p_bar
        self.random_state = random_state

    def get_features(self):
        raise NotImplementedError()

    def get_prediction_result(self) -> PredictionsResult:
        _, result = self.get_features()
        return result



class TrainTestValidator(FeatureSelectionModel):
    def __init__(self, df, features, model, test_size=15, metric='roc-auc', **kwargs):
        super().__init__(df, features, model, **kwargs)
        self.test_size = test_size
        self.metric = metric

    def get_features(self):
        np.random.seed(self.random_state)
        train_idx, test_idx = train_test_split(self.df.index, test_size=self.test_size, stratify=self.df['target'], random_state=self.random_state)

        if self.verbose:
            print('Data split.')
            print('train indices:', np.sort(train_idx.values), 
                'train indices:', np.sort(test_idx.values), sep='\n')
            print()

        new_features, _, _ = select_features(self.df.loc[train_idx], self.features, self.model, self.metric, n_repeats=5, verbose=self.verbose, p_bar=self.p_bar)
        X_1, y_1 = get_x_y(self.df.loc[train_idx], new_features)
        X_2, y_2 = get_x_y(self.df.loc[test_idx], new_features)
        return new_features, train_test(X_1, y_1, X_2, y_2, self.model)




class NestedCrossValidator(FeatureSelectionModel):
    def __init__(self, df, features, model, n_splits=10, n_repeats=10, metric='roc-auc', **kwargs):
        super().__init__(df, features, model, **kwargs)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.metric = metric

    def get_features(self):
        np.random.seed(self.random_state)
        y_preds = []
        y_true = []
        features = []
        cv = KFold(n_splits=self.n_splits, shuffle=True)
        for train_idx, test_idx in get_tqdm_iter(cv.split(self.df), self.p_bar, total=self.n_splits):

            if self.verbose:
                print('Data split.')
                print('train indices:', np.sort(train_idx), 
                    'train indices:', np.sort(test_idx), sep='\n')
                print()

            new_features, _, _, = select_features(self.df.loc[train_idx], self.features, self.model, self.metric, n_repeats=self.n_repeats, verbose=self.verbose, p_bar=self.p_bar-1)
            X_train, y_train = get_x_y(self.df.loc[train_idx], new_features)
            X_test, y_test = get_x_y(self.df.loc[test_idx], new_features)
            self.model.fit(X_train, y_train)
            features.append(new_features)
            y_preds.append(self.model.predict_proba(X_test)[:, 1])
            y_true.append(y_test)
        return features, CVPredictionsResult(y_true, y_preds)


class MultiSegementTraiTest(TrainTestValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_features(self):
        np.random.seed(self.random_state)
        idx = self.df['fn'].drop_duplicates()
        fn_df = self.df.set_index('fn')
        targets = fn_df[~fn_df.index.duplicated('first')]['target'][idx]
        train_ids, test_ids = train_test_split(idx, test_size=self.test_size, stratify=targets, random_state=self.random_state)
        new_features, _, _ = select_features(fn_df.loc[train_ids], self.features, self.model, self.metric, n_repeats=1, verbose=self.verbose, p_bar=self.p_bar)
        X_1, y_1 = get_x_y(fn_df.loc[train_ids], new_features)
        X_2 = get_x(fn_df.loc[test_ids], new_features)

        self.model.fit(X_1, y_1)
        predicted = pd.Series(self.model.predict_proba(X_2)[:, 1], fn_df.loc[test_ids].index, name='pred').groupby('fn').mean()
        results = pd.concat([targets.loc[test_ids], predicted], axis=1)
        return new_features, PredictionsResult.from_df(results)


def get_repeated_scores(method, n_repeats=10, verbose=False, p_bar=1, random_state=42):
    np.random.seed(random_state)
    method.verbose = verbose
    method.p_bar = p_bar - 1
    scores = []
    random_states = np.random.randint(100000, size=n_repeats)
    for i in get_tqdm_iter(range(n_repeats), p_bar):
        if verbose:
            print(f'Iteration {i}.')
        method.random_state = random_states[i]
        score = method.get_prediction_result()
        scores.append(score)
    return scores
