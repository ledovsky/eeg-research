import pandas as pd
from sklearn.metrics import roc_auc_score


def feat_performance(df, features=None, apply_sort=True):
    rows = []
    for feat in features:
        roc_auc = roc_auc_score(df['target'], df[feat])
        roc_auc = max(roc_auc, 1 - roc_auc)
        d = {
            'feature': feat,
            'roc_auc': roc_auc,
            'mean_difference': df[df['target'] == 1][feat].mean() - df[df['target'] == 0][feat].mean()
        }
        rows.append(d)
    res = pd.DataFrame(rows)
    if apply_sort:
        res = res.sort_values('roc_auc', ascending=False)
    return res


def feat_performance_roc_multidf(dfs, features=None, apply_sort=True):

    def get_scores(feat, neg=False):
        rows = []
        sign = 1 if neg is False else -1
        for i, df in enumerate(dfs):
            rows.append((i + 1, feat, roc_auc_score(df['target'], sign * df[feat].fillna(0))))
        return pd.DataFrame(rows, columns=['dataframe', 'feature', 'roc_auc'])

    def swap(row):
        if row['mean'] < 0.5:
            row['mean'], row['max'], row['min'] = 1 - row['mean'], 1 - row['min'], 1 - row['max']
            row['sign'] = 'neg'
        return row

    if features is None:
        features = [feat for feat in dfs[0] if feat not in ['target', 'fn']]

    one_feat_dfs = []
    for feat in features:
        one_feat_dfs.append(
            get_scores(feat)
        )
    df_feats_performance = pd.concat(one_feat_dfs)
    df_feats_performance = (df_feats_performance
        .groupby('feature')['roc_auc']
        .aggregate(['mean', 'min', 'max'])
    )

    df_feats_performance['sign'] = 'pos'

    df_feats_performance = df_feats_performance.apply(swap, axis=1)

    if apply_sort:
        df_feats_performance = df_feats_performance.sort_values('min', ascending=False)

    return df_feats_performance
