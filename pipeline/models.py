from os.path import exists

import operator

import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from .utils import get_tqdm_iter


def create_or_append(df, path):
    if exists(path):
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False)


def run_exper(X, y, custom_model, outliers_rate=0.1, n_expers=100):

    custom_model.outliers_rate = outliers_rate

    if type(y) == type(pd.Series()):
        y = y.values

    roc_aucs = []
    accs = []
    for _ in range(n_expers):
        random_state = np.random.choice(np.arange(10000))
        custom_model.run_cv(X, y, random_state=random_state)
        roc_auc = custom_model.get_roc_auc()
        acc = custom_model.get_accuracy()
        roc_aucs.append(roc_auc)
        accs.append(acc)
    roc_aucs = np.array(roc_aucs)
    accs = np.array(accs)
    return {
        'roc_auc_mean': roc_aucs.mean(),
        'roc_auc_std': roc_aucs.std(),
        'acc_mean': accs.mean(),
        'acc_std': accs.std()
    }



class BaseModel(object):
    """Aim of this class is to provide fast calculation
    of results and metrics that are needed for each model
    """

    def __init__(self, cv='10fold', *args, **kwargs):
        """
        Args:
            X: 2d numpy or pd.DataFrame
            y: 1d numpy array or pd.Series
            model: sklearn-kind model with predict_proba function
            to_drop: idx to drop during train stage
            cv: loo, 5fold or 10fold
        """

        self.y_pred = None
        self.y_true = None
        self.cv = cv

    def _get_cv(self, random_state=42):

        if self.cv == 'loo':
            return LeaveOneOut()
        elif self.cv == '5fold':
            return KFold(n_splits=5, shuffle=True, random_state=random_state)
        elif self.cv == '10fold':
            return KFold(n_splits=10, shuffle=True, random_state=random_state)
        else:
            raise ValueError('wrong value of the cv parameter')

    def run_cv(self, X, y, to_drop=[], random_state=42):
        raise NotImplementedError

    def _set_random_state(self, random_state):
        raise NotImplementedError

    def get_roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred)

    def get_accuracy(self):
        return accuracy_score(self.y_true, self.y_pred > 0.5)

    def get_outliers_idx(self, n_outliers=5):
        outliers_idx = (
            np.argsort(
                np.abs(self.y_pred - self.y_true))
            [::-1]
            [:n_outliers])
        return outliers_idx

    def get_ypred_df(self):
        df = pd.DataFrame({'y_true': self.y_true, 'y_pred': self.y_pred})
        return df


class RegularModel(BaseModel):
    def __init__(self, model, *args, **kwargs):
        super(RegularModel, self).__init__(*args, **kwargs)
        self.model = model

    def run_cv(self, X, y, to_drop=[], random_state=42):
        self.model.set_params(random_state=random_state)
        self.y_pred = np.zeros(shape=y.shape)
        self.y_true = y
        cv = self._get_cv(random_state=random_state)
        for train_idx, test_idx in cv.split(X, y):
            if len(to_drop):
                train_idx = [idx for idx in train_idx if idx not in to_drop]
            self.model.fit(X[train_idx, :], y[train_idx])
            self.y_pred[test_idx] = self.model.predict_proba(X[test_idx, :])[:, 1]


class NoOutliersModel(BaseModel):
    def __init__(self, model_1, model_2, outliers_rate=0.10, *args, **kwargs):
        super(NoOutliersModel, self).__init__(*args, **kwargs)
        self.outliers_rate = outliers_rate
        self.model_1 = model_1
        self.model_2 = model_2

    def run_cv(self, X, y, random_state=42, *args, **kwargs):
        self.model_1.set_params(random_state=random_state)
        self.model_2.set_params(random_state=random_state)
        submodel_1 =  RegularModel(self.model_1, random_state=random_state)
        submodel_2 = RegularModel(self.model_2, random_state=random_state)
        submodel_1.run_cv(X, y, random_state=random_state)
        n_outliers = int(self.outliers_rate * len(X))
        outliers_idx = submodel_1.get_outliers_idx(n_outliers=n_outliers)
        submodel_2.run_cv(X, y, to_drop=outliers_idx, random_state=random_state)

        self.y_true = submodel_2.y_true
        self.y_pred = submodel_2.y_pred


class LRModel(RegularModel):
    def __init__(self, *args, **kwargs):
        model = LogisticRegression(solver='liblinear')
        super(LRModel, self).__init__(model=model, *args, **kwargs)

    def run_cv(self, X, y, *args, **kwargs):
        X = StandardScaler().fit_transform(X)
        super(LRModel, self).run_cv(X, y, *args, **kwargs)


class LRModelNoOut(NoOutliersModel):
    def __init__(self, *args, **kwargs):
        model_1 = LogisticRegression(solver='liblinear')
        model_2 = LogisticRegression(solver='liblinear')
        super(LRModelNoOut, self).__init__(model_1=model_1, model_2=model_2, *args, **kwargs)

    def run_cv(self, X, y, *args, **kwargs):
        X = StandardScaler().fit_transform(X)
        super(LRModelNoOut, self).run_cv(X, y, *args, **kwargs)


class TreeModel(RegularModel):
    def __init__(self, *args, **kwargs):
        model = DecisionTreeClassifier()
        super(TreeModel, self).__init__(model=model, *args, **kwargs)


class XGBModel(RegularModel):
    def __init__(self, *args, **kwargs):
        model = xgb.XGBClassifier(max_depth=4, n_estimators=30, n_jobs=4)
        super(XGBModel, self).__init__(model=model, *args, **kwargs)


class XGBModelNoOut(NoOutliersModel):
    def __init__(self, *args, **kwargs):
        model_1 = xgb.XGBClassifier(max_depth=4, n_estimators=30)
        model_2 = xgb.XGBClassifier(max_depth=4, n_estimators=30)
        super(XGBModelNoOut, self).__init__(model_1=model_1, model_2=model_2, *args, **kwargs)


class RFModel(RegularModel):
    def __init__(self, *args, **kwargs):
        model = RandomForestClassifier(n_estimators=30, max_depth=2)
        super(RFModel, self).__init__(model=model, *args, **kwargs)


from sklearn.base import BaseEstimator


class LRScaled(BaseEstimator):
    def __init__(self, random_state=42, **kwargs):
        self.transformer = StandardScaler()
        self.lr = LogisticRegression(random_state=random_state, solver='liblinear', **kwargs)

    def fit(self, X, y):
        StandardScaler().fit_transform(X)
        X = self.transformer.fit_transform(X)
        self.lr.fit(X, y)
        return self

    def set_params(self, **params):
        if 'random_state' in params:
            del params['random_state']
        return super().set_params(**params)

    def predict(self, X):
        X = self.transformer.transform(X)
        return self.lr.predict(X)

    def predict_proba(self, X):
        X = self.transformer.transform(X)
        return self.lr.predict_proba(X)

class SVMScaled(BaseEstimator):
    def __init__(self, random_state=42, **kwargs):
        self.transformer = StandardScaler()
        self.model = SVC(kernel='linear', random_state=random_state, probability=True)

    def fit(self, X, y):
        X = self.transformer.fit_transform(X, y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.transformer.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = self.transformer.transform(X)
        return self.model.predict_proba(X)


class PredictionsResult:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true.copy()
        self.y_pred = y_pred.copy()

    @classmethod
    def from_df(PredictionsResult, df):
        return PredictionsResult(df['target'], df['pred'])

    @property
    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred)

    @property
    def acc(self):
        return accuracy_score(self.y_true, self.y_pred > 0.5)


class CVPredictionsResult:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true.copy()
        self.y_pred = y_pred.copy()
        self._roc_aucs = None
        self._accs = None

    @property
    def roc_aucs(self):
        if self._roc_aucs is None:
            self._roc_aucs = []
            for y_true, y_pred in zip(self.y_true, self.y_pred):
                self._roc_aucs.append(roc_auc_score(y_true, y_pred))
            self._roc_aucs = np.array(self._roc_aucs)
        return self._roc_aucs

    @property
    def accs(self):
        if self._accs is None:
            self._accs = []
            for y_true, y_pred in zip(self.y_true, self.y_pred):
                self._accs.append(accuracy_score(y_true, y_pred > .5))
            self._accs = np.array(self._accs)
        return self._accs


class RepeatedPredictionsResult:
    def __init__(self, y_true, y_preds):

        assert len(y_true.shape) == 1
        assert len(y_preds.shape) == 2

        self.y_true = y_true.copy()
        self.y_preds = y_preds.copy()
        self._roc_aucs = None
        self._accs = None

    @property
    def roc_aucs(self):
        if self._roc_aucs is None:
            self._roc_aucs = []
            for i in range(self.y_preds.shape[1]):
                self._roc_aucs.append(roc_auc_score(self.y_true, self.y_preds[:, i]))
            self._roc_aucs = np.array(self._roc_aucs)
        return self._roc_aucs

    @property
    def accs(self):
        if self._accs is None:
            self._accs = []
            for i in range(self.y_preds.shape[1]):
                self._accs.append(accuracy_score(self.y_true, self.y_preds[:, i] > 0.5))
            self._accs = np.array(self._accs)
        return self._accs


def kfold(X, y, model, n_splits=10, random_state=42, to_drop=None):
    """

    Args:
        X:
        y:
        model:
        to_drop:
        n_splits:
        random_state:

    Returns:
        res: PredictionsResult
    """

    # model.set_params(random_state=random_state)
    y_pred = np.zeros(shape=y.shape)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in cv.split(X, y):
        # TODO: check how to_drop works
        if to_drop:
            train_idx = [idx for idx in train_idx if idx not in to_drop]

        model.fit(X[train_idx, :], y[train_idx])
        y_pred[test_idx] = model.predict_proba(X[test_idx, :])[:, 1]

    res = PredictionsResult(y, y_pred)
    return res


def repeated_kfold(X, y, model, n_splits=10, n_repeats=100, random_state=42):
    """

    Args:
        X:
        y:
        model:
        n_splits:
        n_repeats:
        random_state:

    Returns:
        res:
    """

    np.random.seed(random_state)
    random_states = np.random.randint(100000, size=(n_repeats))

    y_preds = np.empty(shape=(y.shape[0], n_repeats))

    for i in range(n_repeats):
        kfold_res = kfold(X, y, model, n_splits=n_splits, random_state=random_states[i])
        y_preds[:, i] = kfold_res.y_pred

    res = RepeatedPredictionsResult(y, y_preds)
    return res


def select_features(df, features, model, metric='roc-auc', df_val=None, n_repeats=20,
                    threshold=0.005, take_first=None, order_func=None, verbose=False,
                    p_bar=1):
    X = df[features].fillna(0).values
    y = df['target'].values
    if df_val is not None:
        y_val = df_val['target'].values
    def get_metric(data):
        if metric == 'roc-auc':
            return data.roc_aucs
        if metric == 'acc':
            return data.accs

    if order_func is None:
        X_sc = StandardScaler().fit_transform(X)
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_sc, y)
        weights = [(f, np.abs(w)) for f, w in zip(features, lr.coef_[0])]
        weights = sorted(weights, key=operator.itemgetter(1))
        features = list(list(zip(*weights))[0])[::-1]
    else:
        features = order_func(features, df)

    if take_first:
        features = features[:take_first]

    new_features = []

    best_res = None
    hist = []


    def update(new_features, cur_features, best_res, hist, action):
        def condition(res, best_res):
            if best_res is None:
                if get_metric(res).mean() > 0.5 + threshold:
                    return True
                else:
                    return False
            elif get_metric(res).mean() - get_metric(best_res).mean() > threshold:
                return True
            else:
                return False
        X = df[cur_features].fillna(0).values
        res = repeated_kfold(X, y, model, n_splits=10, n_repeats=n_repeats)
        if verbose:
            print(f'{res.roc_aucs.mean():4f}\t{res.accs.mean():4f}')
            pass
        if condition(res, best_res):
            new_features = cur_features
            best_res = res
            d = {
                'feature': feature,
                'action': action,
                'score': get_metric(res).mean(),
                'score_std': get_metric(res).std()
            }
            if df_val is not None:
                X = df_val[cur_features].fillna(0).values
                val_res = repeated_kfold(X, y_val, model, n_splits=10, n_repeats=n_repeats)
                d['score_val'] = get_metric(val_res).mean()
                d['score_val_std'] = get_metric(val_res).std()
            hist.append(d)
        return new_features, best_res, hist


    if verbose:
        print('Feature selection. Step 1')
        print('feature_name    \troc-auc \taccuracy')

    for feature in get_tqdm_iter(features, p_bar):
        if verbose:
            print(f'{feature:16}', end='\t')
        cur_features = new_features + [feature]
        new_features, best_res, hist = update(new_features, cur_features, best_res, hist, action='added')

    if verbose:
        print()
        print('Feature selection. Step 2')
        print('feature_name    \troc-auc \taccuracy')

    features = new_features.copy()
    if len(features) > 1:
        for feature in get_tqdm_iter(features, p_bar):
            if verbose:
                print(f'{feature:16}', end='\t')
            cur_features = [_f for _f in new_features if _f != feature]
            new_features, best_res, hist = update(new_features, cur_features, best_res, hist, action='removed')

    if verbose:
        print()

    return new_features, best_res, pd.DataFrame(hist)


def train_test(X_1, y_1, X_2, y_2, model):
    model.fit(X_1, y_1)
    y_pred = model.predict_proba(X_2)[:, 1]
    return PredictionsResult(y_2, y_pred)
