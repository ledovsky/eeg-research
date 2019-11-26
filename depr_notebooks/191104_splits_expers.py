import sys
from os.path import join

import pandas as pd

sys.path.append('../pipeline')

from models import LRScaled, repeated_kfold, train_test, get_x_y
from features import get_feature_path, get_merged_df


splits_data = __import__('191104_splits_calc_feats')


features = [
    'coh_alpha_fp1_fz',
    'coh_beta_fp1_fz',
    'coh_beta_f3_f4',
    'coh_beta_o1_t5',
    # 'bands_gamma_fz',
    'coh_alpha_t4_f8',
    'env_alpha_f3_t5',
    'coh_beta_f7_fz',
    'env_alpha_t4_f3',
    'coh_beta_o2_fz',
    'env_alpha_c4_o2',
    'env_alpha_t6_fz',
    'coh_alpha_o1_f3',
    'coh_beta_t4_f8',
    # 'bands_gamma_t5',
    'coh_beta_f4_fp1',
    'coh_beta_c3_p3',
    # 'bands_gamma_p4',
    'env_beta_p4_c4',
    'coh_alpha_o2_c3',
    'env_alpha_f8_pz',
    'env_beta_c3_t5']


feature_methods = [
    'env-alpha',
    'env-beta',
    'coh-alpha',
    'coh-beta',
    'bands'
]

model = LRScaled()

base_path = '../../own_data/191104_depr_splits_features/'


if __name__ == '__main__':

    dir_1 = '60s_0_0_60'
    dir_2 = '60s_1_0_60'

    df_1 = get_merged_df(join(base_path, dir_1), feature_methods)
    df_2 = get_merged_df(join(base_path, dir_2), feature_methods)
    X_1, y_1 = get_x_y(df_1, features)
    X_2, y_2 = get_x_y(df_2, features)
    res_1 = repeated_kfold(X_1, y_1, model, n_repeats=100)
    res_2 = repeated_kfold(X_2, y_2, model, n_repeats=100)
    print('In-split quality. ROC-AUC split 1 = {:.2f}, split 2 = {:.2f}'
          .format(res_1.roc_aucs.mean(), res_2.roc_aucs.mean()))

    res_12 = train_test(X_1, y_1, X_2, y_2, model)
    res_21 = train_test(X_2, y_2, X_1, y_1, model)

    print('Cross split quality. ROC-AUC 1->2 {:.2f} 2->1 {:.2f}'
          .format(res_12.roc_auc, res_21.roc_auc))




