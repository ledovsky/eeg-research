from os.path import exists, join
from os import mkdir, remove, listdir

import operator

import pandas as pd
import numpy as np

from tqdm import trange, tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


def create_or_append(df, path):
    if exists(path):
        df.to_csv(path, mode='a', index=False, header=False)
    else:
        df.to_csv(path, index=False)


def get_model_func(method):
    do_feature_selection = False
    if method == 'lr':
        custom_model = LRModel()
    elif method == 'lr-feat-sel':
        custom_model = LRModel()
        do_feature_selection = True
    elif method == 'xgb':
        custom_model = XGBModel()
    elif method == 'xgb-feat-sel':
        custom_model = XGBModel()
        do_feature_selection = True
    elif method == 'rf':
        custom_model = RFModel()
    elif method == 'rf-feat-sel':
        custom_model = RFModel()
        do_feature_selection = True
    elif method == 'cart':
        custom_model = TreeModel()
    elif method == 'cart-feat-sel':
        custom_model = TreeModel()
        do_feature_selection = True
    else:
        raise Exception('Method is not in list of allowed methods - ', method)

    def wrapped(data_path, out_path):

        print('Started model stage -', method)

        features_path = join(out_path, 'features')

        models_res_dir = join(out_path, 'models_predict_samples')
        if not exists(models_res_dir):
            mkdir(models_res_dir)

        res_path = join(out_path, 'results.csv')

        for fn in listdir(features_path):
            if fn[-3:] != 'csv':
                continue
            path = join(features_path, fn)

            df = pd.read_csv(path)
            features = [col for col in df.columns if col not in ['dataset', 'fn', 'target']]
            if do_feature_selection:
                features = select_features(df, features, custom_model)
            X = df[features].fillna(0).values
            y = df['target'].values
            res = run_exper(X, y, custom_model)
            res['model'] = method
            res['features'] = fn[:-4]
            create_or_append(pd.DataFrame([res]), res_path)

            # save to sample predictions
            def get_pred_df():
                random_state = np.random.choice(np.arange(10000))
                custom_model.run_cv(X, y, random_state=random_state)
                prediction_df = pd.DataFrame({'y_pred': custom_model.y_pred, 'y_true': custom_model.y_true})
                return prediction_df

            prediction_df = get_pred_df()
            prediction_df.to_csv(join(models_res_dir, method.replace('-', '_') + '_' + fn[:-4] + '.csv'), index=False)

            prediction_df = get_pred_df()
            prediction_df.to_csv(join(models_res_dir, method.replace('-', '_') + '_' + fn[:-4] + '_2.csv'), index=False)

    return wrapped


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


def select_features(df, features, custom_model):
    X = df[features].fillna(0).values
    y = df['target'].values

    X_sc = StandardScaler().fit_transform(X)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_sc, y)
    weights = [(f, np.abs(w)) for f, w in zip(features, lr.coef_[0])]
    weights = sorted(weights, key=operator.itemgetter(1))
    features = list(list(zip(*weights))[0])[::-1]

    new_features = []

    best_score = None

    print('Feature selection. Step 1')

    for f in tqdm(features):

        cur_features = new_features + [f]

        X = df[cur_features].fillna(0).values
        cur_score = run_exper(X, y, custom_model, n_expers=20)['roc_auc_mean']

        if best_score is None or (cur_score - best_score) > 0.005:
            new_features = cur_features
            best_score = cur_score

    print('Feature selection. Step 2')
    features = new_features.copy()

    for f in tqdm(features):
        cur_features = [_f for _f in new_features if _f != f]

        X = df[cur_features].fillna(0).values
        cur_score = run_exper(X, y, custom_model, n_expers=20)['roc_auc_mean']

        if best_score is None or (cur_score - best_score) > 0.005:
            new_features = cur_features
            best_score = cur_score

    return new_features


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
