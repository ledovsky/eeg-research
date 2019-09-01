from os.path import isfile, exists, join
from shutil import rmtree
from os import mkdir, remove, listdir

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


def create_or_append(df, path):
    if exists(path):
        df.to_csv(path, mode='a', index=False)
    else:
        df.to_csv(path, index=False)


def get_model_func(method):
    if method == 'lr':
        custom_model = LRModel()
    elif method == 'lr-no-out':
        custom_model = LRModelNoOut()
    elif method == 'xgb':
        custom_model = XGBModel()
    elif method == 'xgb-no-out':
        custom_model = XGBModelNoOut()
    elif method == 'rf':
        custom_model = RFModel()
    elif method == 'cart':
        custom_model = TreeModel()
    else:
        raise Exception('Method is not in list of allowed methods - ', method)

    def wrapped(data_path, out_path):
        features_path = join(out_path, 'features')
        res_path = join(out_path, 'results.csv')
        for fn in listdir(features_path):
            if fn[-3:] != 'csv':
                continue
            path = join(features_path, fn)

            df = pd.read_csv(path)
            features = [col for col in df.columns if col not in ['dataset', 'fn', 'is_patient']]
            X = df[features].fillna(0).values
            y = df['is_patient'].values
            res = run_exper(X, y, custom_model)
            res['model'] = method
            create_or_append(pd.DataFrame([res]), res_path)

    return wrapped


def run_exper(X, y, custom_model, n_outliers=10):

    n_expers = 100

    if type(y) == type(pd.Series()):
        y = y.values

    roc_aucs = []
    accs = []
    for _ in range(n_expers):
        random_state = np.random.choice(np.arange(1000))
        custom_model.run_cv()
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

    def __init__(self, model, to_drop=[], cv='loo'):
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

        if cv == 'loo':
            self.cv = LeaveOneOut()
        elif cv == '5fold':
            # self.cv = StratifiedKFold(n_splits=5)
            self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        elif cv == '10fold':
            # self.cv = StratifiedKFold(n_splits=10)
            self.cv = KFold(n_splits=10, shuffle=True, random_state=42)
        else:
            raise ValueError('wrong value of the cv parameter')
        self.model = model

    def run_cv(self, X, y, to_drop=[]):
        self.y_pred = np.zeros(shape=y.shape)
        self.y_true = y
        for train_idx, test_idx in self.cv.split(X, y):
            if to_drop:
                train_idx = [idx for idx in train_idx if idx not in to_drop]
            self.model.fit(X[train_idx, :], y[train_idx])
            self.y_pred[test_idx] = self.model.predict_proba(X[test_idx, :])[:, 1]

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


class NoOutliersModel(BaseModel):
    def __init__(self, model_1, model_2, n_outliers=10, *args, **kwargs):
        super(NoOutliersModel, self).init(*args, **kwargs)
        self.n_outliers = n_outliers
        self.model_1 = model_1
        self.model_2 = model_2

    def run_cv(self, X, y):
        submodel_1 =  BaseModel(self.model_1)
        submodel_2 = BaseModel(self.model_2)
        submodel_1.run_cv(X, y)
        outliers_idx = submodel_1.get_outliers_idx(n_outliers=self.n_outliers)
        submodel_2.run_cv(X, y, to_drop=outliers_idx)

        self.y_true = submodel_2.y_true
        self.y_pred = submodel_2.y_pred


class LRModel(BaseModel):
    def __init__(self, *args, **kwargs):
        model = LogisticRegression(solver='liblinear')
        super(LRModel, self).init(model=model, *args, **kwargs)

    def run_cv(self, X, y, to_drop=[]):
        X = StandardScaler().fit_transform(X)
        super(LRModel, self).run_cv(X, y, to_drop)


class LRModelNoOut(NoOutliersModel):
    def __init__(self, *args, **kwargs):
        model_1 = LogisticRegression(solver='liblinear')
        model_2 = LogisticRegression(solver='liblinear')
        super(LRModelNoOut, self).init(model_1=model_1, model_2=model_2, *args, **kwargs)

    def run_cv(self, X, y, to_drop=[]):
        X = StandardScaler().fit_transform(X)
        super(LRModelNoOut, self).run_cv(X, y, to_drop)


class TreeModel(BaseModel):
    def __init__(self, *args, **kwargs):
        model = DecisionTreeClassifier()
        super(TreeModel, self).init(model=model, *args, **kwargs)


class XGBModel(BaseModel):
    def __init__(self, *args, **kwargs):
        model = xgb.XGBClassifier(max_depth=4, n_estimators=30)
        super(XGBModel, self).init(model=model, *args, **kwargs)


class XGBModelNoOut(BaseModel):
    def __init__(self, *args, **kwargs):
        model_1 = xgb.XGBClassifier(max_depth=4, n_estimators=30)
        model_2 = xgb.XGBClassifier(max_depth=4, n_estimators=30)
        super(XGBModelNoOut, self).init(model_1=model_1, model_2=model_2, *args, **kwargs)


class RFModel(BaseModel):
    def __init__(self, *args, **kwargs):
        model = RandomForestClassifier(n_estimators=30, max_depth=2)
        super(RFModel, self).init(model=model, *args, **kwargs)
