import sys
from shutil import rmtree
from os import mkdir
from os.path import join

sys.path.append('../pipeline')

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from models import LRScaled, select_features
from features import get_feature_path


models_to_test = [
    ('lr', LRScaled()),
    ('xgb-4', xgb.XGBClassifier(max_depth=4, n_estimators=30, n_jobs=4)),
    ('xgb-2', xgb.XGBClassifier(max_depth=2, n_estimators=30, n_jobs=4)),
    ('cart', DecisionTreeClassifier()),
]

features_to_test = [
    'coh-alpha',
    'coh-beta',
    'env-alpha',
    'env-beta',
    'bands'
]


if __name__ == '__main__':

    data_path = '../../own_data/191104_depr_splits_features/60s_1_0_60'
    out_path = '../../own_data/191122_mdd_article'
    clear = True


    print('Started grid pipeline')
    # if specified take CLI param else look at config

    if clear:
        rmtree(out_path)
        mkdir(out_path)

    rows = []
    for model_name, model in models_to_test:
        for features_name in features_to_test:

            print(model_name, features_name)

            df = pd.read_csv(get_feature_path(features_name, data_path))

            features = [col for col in df.columns if col not in ['dataset', 'fn', 'target']]

            features_selected, result, hist = select_features(df, features, model)

            d = {
                'model': model_name,
                'features': features_name,
                'roc_auc_mean': result.roc_aucs.mean(),
                'roc_auc_std': result.roc_aucs.std(),
                'acc_mean': result.accs.mean(),
                'acc_std': result.accs.std(),
            }
            rows.append(d)

    pd.DataFrame(rows).to_csv(join(out_path, 'result.csv'))

